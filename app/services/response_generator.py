"""
Response Generator - Generate natural responses in multiple languages
"""
import re
from typing import Dict, Optional, List, Tuple
from app.services.intent_classifier import IntentClassifier
from app.services.knowledge_engine import KnowledgeEngine
from app.services.escalation_manager import EscalationManager
from app.services.followup_generator import FollowupGenerator
from app.services.proactive_suggestions import ProactiveSuggestions
from app.services.ivf_guardrail import IVFGuardrail
from app.models.schemas import BotResponse, SuggestedAction, RelatedContent
from app.utils.translator import translation_service
from app.config import settings
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generate bot responses based on intent and context"""
    
    # Response templates in English
    TEMPLATES_EN = {
        'greeting': "Hello! I'm here to help answer your questions about IVF treatment. What would you like to know?",
        'goodbye': "Thank you for chatting! If you have more questions, feel free to ask anytime. Take care!",
        'no_results': "I'm sorry, I couldn't find specific information about that. Would you like me to connect you with a counsellor who can help?",
        'appointment_schedule': "I'd be happy to help you schedule an appointment. What date works best for you?",
        'appointment_reschedule': "I can help you reschedule your appointment. Please let me know your preferred date and time.",
        'escalation_acknowledgment': "I understand your concern. I'm connecting you with a counsellor who can better assist you. They'll be with you shortly.",
        'low_confidence': "I want to make sure I give you the right information. Could you provide a bit more detail about what you're looking for?"
    }
    
    # Response templates in Hindi
    TEMPLATES_HI = {
        'greeting': "नमस्ते! मैं आईवीएफ उपचार के बारे में आपके सवालों के जवाब देने में मदद करने के लिए यहां हूं। आप क्या जानना चाहेंगे?",
        'goodbye': "बातचीत के लिए धन्यवाद! यदि आपके और प्रश्न हैं, तो कृपया कभी भी पूछें। अपना ख्याल रखें!",
        'no_results': "मुझे खेद है, मुझे उसके बारे में विशिष्ट जानकारी नहीं मिली। क्या आप चाहेंगे कि मैं आपको एक काउंसलर से जोड़ूं जो मदद कर सकता है?",
        'appointment_schedule': "मैं आपकी अपॉइंटमेंट शेड्यूल करने में मदद करने के लिए खुश हूं। आपके लिए कौन सी तारीख सबसे अच्छी है?",
        'appointment_reschedule': "मैं आपकी अपॉइंटमेंट को पुनर्निर्धारित करने में मदद कर सकता हूं। कृपया मुझे अपनी पसंदीदा तारीख और समय बताएं।",
        'escalation_acknowledgment': "मैं आपकी चिंता समझता हूं। मैं आपको एक काउंसलर से जोड़ रहा हूं जो आपकी बेहतर सहायता कर सकता है। वे जल्द ही आपके साथ होंगे।",
        'low_confidence': "मैं यह सुनिश्चित करना चाहता हूं कि मैं आपको सही जानकारी दूं। क्या आप मुझे बता सकते हैं कि आप क्या खोज रहे हैं?"
    }
    
    def __init__(
        self,
        intent_classifier: IntentClassifier,
        knowledge_engine: KnowledgeEngine,
        escalation_manager: EscalationManager
    ):
        self.intent_classifier = intent_classifier
        self.knowledge_engine = knowledge_engine
        self.escalation_manager = escalation_manager
        self.followup_generator = FollowupGenerator()
        self.proactive_suggestions = ProactiveSuggestions()
        self.ivf_guardrail = IVFGuardrail()
    
    async def generate_response(
        self,
        message: str,
        conversation_context: Dict,
        language: str = "en"
    ) -> Dict:
        """
        Generate response to user message (with caching for performance)
        """
        # Build contextualized query early so we can skip cache when context was used
        history = conversation_context.get('history', [])
        query_for_search_and_intent = self._get_contextualized_query(message, history, language)
        use_context = query_for_search_and_intent.strip() != (message or '').strip()
        # Check full response cache only when query was not contextualized (same response for same message)
        if not use_context and hasattr(self.knowledge_engine, 'redis_client') and self.knowledge_engine.redis_client:
            try:
                cache_key = f"response:{hashlib.md5(f'{message}:{language}'.encode()).hexdigest()}"
                cached = self.knowledge_engine.redis_client.get(cache_key)
                if cached:
                    logger.debug(f"Full response cache hit: {message[:50]}")
                    return json.loads(cached)
            except Exception as e:
                logger.debug(f"Cache check failed: {e}")
        
        # GUARDRAIL: Check if query is IVF-related before processing
        is_ivf_related, ivf_confidence, rejection_reason = self.ivf_guardrail.is_ivf_related(
            message, language, conversation_context
        )
        
        if not is_ivf_related:
            # Query is not IVF-related - return rejection message
            logger.info(f"Non-IVF query rejected: {message[:50]} (reason: {rejection_reason})")
            rejection_text = self.ivf_guardrail.get_rejection_message(rejection_reason, language)
            
            # Add suggested IVF-related actions
            suggested_actions = self._get_default_suggestions(language)
            # Always add HomeIVF link
            homeivf_action = self._get_homeivf_link_action(language)
            if not any(action.url == settings.HOMEIVF_WEBSITE_URL for action in suggested_actions):
                suggested_actions.append(homeivf_action)
            suggested_actions = self._filter_used_suggestions(
                suggested_actions, conversation_context.get('history', [])
            )
            bot_response = BotResponse(
                text=rejection_text,
                type='text',
                language=language,
                suggested_actions=suggested_actions,
                related_content=[],
                confidence=0.0
            )
            
            return {
                'response': bot_response,
                'intent': {
                    'intent': 'non_ivf_query',
                    'confidence': 0.0,
                    'all_probabilities': {'non_ivf_query': 1.0},
                    'language': language
                },
                'escalation': {'escalate': False}
            }
        
        """
        Generate response to user message
        
        Args:
            message: User message
            conversation_context: Conversation context (history, patient info, etc.)
            language: Language code ('en' or 'hi')
        
        Returns:
            Dict with response, intent, confidence, and suggested actions
        """
        # Parallelize intent classification and escalation check for faster response
        import asyncio
        intent_task = self.intent_classifier.classify(query_for_search_and_intent, language, conversation_context)
        # Start escalation check early (will use intent later if needed)
        # Can still detect urgent keywords and emotional indicators without intent
        escalation_task = self.escalation_manager.assess_escalation_need(
            message, None, conversation_context.get('history', []), language
        )
        
        # Wait for both to complete in parallel
        intent_result, escalation_result = await asyncio.gather(intent_task, escalation_task)
        intent = intent_result['intent']
        confidence = intent_result['confidence']
        
        # Update escalation with intent if needed (only if first check didn't escalate)
        if not escalation_result.get('escalate'):
            escalation_result = await self.escalation_manager.assess_escalation_need(
                message, intent, conversation_context.get('history', []), language
            )
        
        response_text = ""
        suggested_actions = []
        related_content = []
        
        # Handle different intents
        if intent == 'greeting':
            response_text = self._get_template('greeting', language)
            suggested_actions = self._get_default_suggestions(language)
        
        elif intent == 'goodbye':
            response_text = self._get_template('goodbye', language)
        
        elif intent.startswith('faq_'):
            # When user picks a topic phrase (e.g. "Lifestyle & Diet", "जीवनशैली और आहार"), filter by category
            category_for_search = self._get_category_for_topic_query(message, language)
            # PRIMARY: Search knowledge base with conversation context for follow-up questions
            search_results = self.knowledge_engine.search(
                query_for_search_and_intent,
                language=language,
                top_k=settings.TOP_K_SEARCH_RESULTS,
                category=category_for_search,
            )
            
            # Check if we have a good match from FAQs
            best_similarity = search_results[0]['similarity'] if search_results else 0.0
            logger.debug(f"FAQ search: {len(search_results)} results (best: {best_similarity:.2f}, threshold: {settings.MIN_CONFIDENCE_SCORE})")
            
            if search_results and best_similarity >= settings.MIN_CONFIDENCE_SCORE:
                # Use FAQ result (optimized path)
                top_result = search_results[0]
                response_text = top_result.get('answer', top_result.get('content', ''))
                category = top_result.get('category') or 'general'
                response_lang = self._response_language(response_text, language)
                
                # First: suggestion chips from FAQs (other search results)
                faq_chips = self._suggested_actions_from_faq_results(
                    search_results, skip_index=1, max_chips=4
                )
                suggested_actions.extend(faq_chips)
                # Then: template follow-ups, excluding questions already shown as FAQ chips
                if settings.ENABLE_FOLLOWUPS:
                    exclude = [a.label for a in faq_chips]
                    followups = self.followup_generator.generate_followups(
                        category=category,
                        language=response_lang,
                        answer=response_text,
                        exclude_questions=exclude if exclude else None
                    )
                    for followup in followups:
                        suggested_actions.append(SuggestedAction(
                            type='quick_reply',
                            label=followup,
                            action='faq_followup',
                            data={'question': followup}
                        ))
                
                # Add proactive suggestions (same language as answer)
                if settings.ENABLE_PROACTIVE_SUGGESTIONS:
                    proactive = self.proactive_suggestions.get_suggestions(
                        current_topic=category,
                        language=response_lang,
                        conversation_history=conversation_context.get('history', [])
                    )
                    suggested_actions.extend(proactive)
                
                # Add related content
                for result in search_results[1:3]:
                    related_content.append(RelatedContent(
                        title=result.get('title', result.get('question', '')),
                        id=result.get('article_id', result.get('faq_id', '')),
                        type=result.get('type', 'article')
                    ))
            else:
                # FALLBACK: Use medgemma if FAQ doesn't have good match (with contextualized query)
                response_text, suggested_actions, related_content = self._handle_medgemma_fallback(
                    message, intent, language, conversation_context, search_results,
                    contextualized_query=query_for_search_and_intent
                )
        
        elif intent == 'appointment_schedule':
            response_text = self._get_template('appointment_schedule', language)
            suggested_actions.append(SuggestedAction(
                type='button',
                label=self._translate_action_label('View Available Slots', language),
                action='view_slots'
            ))
            # Add proactive suggestions
            if settings.ENABLE_PROACTIVE_SUGGESTIONS:
                proactive = self.proactive_suggestions.get_quick_actions(language)
                suggested_actions.extend(proactive[:2])  # Add 2 quick actions
        
        elif intent == 'appointment_reschedule':
            response_text = self._get_template('appointment_reschedule', language)
            suggested_actions.append(SuggestedAction(
                type='button',
                label=self._translate_action_label('Reschedule Appointment', language),
                action='reschedule'
            ))
        
        elif intent.startswith('escalation_') or escalation_result.get('escalate', False):
            response_text = self._get_template('escalation_acknowledgment', language)
            if escalation_result.get('urgency') == 'urgent':
                response_text += f"\n\n{self._get_urgent_message(language)}"
        
        else:
            # Generic response - try FAQ first (with context); use category when user picked a topic phrase
            category_for_search = self._get_category_for_topic_query(message, language)
            search_results = self.knowledge_engine.search(
                query_for_search_and_intent,
                language=language,
                top_k=settings.TOP_K_SEARCH_RESULTS,
                category=category_for_search,
            )
            
            if search_results and search_results[0]['similarity'] >= settings.MIN_CONFIDENCE_SCORE:
                response_text = search_results[0].get('answer', search_results[0].get('content', ''))
                category = search_results[0].get('category') or 'general'
                response_lang = self._response_language(response_text, language)
                
                # First: suggestion chips from FAQs (other search results)
                faq_chips = self._suggested_actions_from_faq_results(
                    search_results, skip_index=1, max_chips=4
                )
                suggested_actions.extend(faq_chips)
                exclude = [a.label for a in faq_chips]
                # Then: template follow-ups
                if settings.ENABLE_FOLLOWUPS:
                    followups = self.followup_generator.generate_followups(
                        category=category,
                        language=response_lang,
                        answer=response_text,
                        exclude_questions=exclude if exclude else None
                    )
                    for followup in followups[:2]:  # Limit to 2 for generic responses
                        suggested_actions.append(SuggestedAction(
                            type='quick_reply',
                            label=followup,
                            action='faq_followup',
                            data={'question': followup}
                        ))
            else:
                # Try medgemma as fallback (reuse common method)
                response_text, new_actions, new_content = self._handle_medgemma_fallback(
                    message, intent, language, conversation_context, search_results, max_followups=2,
                    contextualized_query=query_for_search_and_intent
                )
                suggested_actions.extend(new_actions)
                related_content.extend(new_content)
                
                if not response_text:
                    if confidence < 0.6:
                        response_text = self._get_template('low_confidence', language)
                    else:
                        response_text = self._get_template('no_results', language)
        
        # Normalize response text (fix list numbering, abbreviations, extra newlines from AI)
        response_text = self._normalize_response_text(response_text or '')
        # Always add HomeIVF link to Actions section (use answer language when answer is in Hindi)
        effective_language = self._response_language(response_text or '', language)
        homeivf_action = self._get_homeivf_link_action(effective_language)
        # Check if HomeIVF link is not already in suggested_actions
        if not any(action.url == settings.HOMEIVF_WEBSITE_URL for action in suggested_actions):
            suggested_actions.append(homeivf_action)
        # Remove suggestion chips that the user has already used (appear in conversation history)
        suggested_actions = self._filter_used_suggestions(
            suggested_actions, conversation_context.get('history', [])
        )
        # Create response object (language matches answer so UI is monolingual)
        bot_response = BotResponse(
            text=response_text,
            type='text',
            language=effective_language,
            suggested_actions=suggested_actions,
            related_content=related_content,
            confidence=confidence
        )
        
        result = {
            'response': bot_response,
            'intent': intent_result,
            'escalation': escalation_result
        }
        
        # Cache full response for common queries (cache for 1 hour)
        # Only cache when query was not contextualized (context-dependent responses vary by history)
        if not use_context and hasattr(self.knowledge_engine, 'redis_client') and self.knowledge_engine.redis_client:
            try:
                if confidence >= settings.MIN_CONFIDENCE_SCORE and intent.startswith('faq_'):
                    cache_key = f"response:{hashlib.md5(f'{message}:{language}'.encode()).hexdigest()}"
                    self.knowledge_engine.redis_client.setex(
                        cache_key,
                        3600,  # 1 hour cache
                        json.dumps(result, default=str)
                    )
            except Exception as e:
                logger.debug(f"Response cache write failed: {e}")
        
        return result
    
    def _get_template(self, template_key: str, language: str) -> str:
        """Get response template in specified language"""
        templates = self.TEMPLATES_HI if language == 'hi' else self.TEMPLATES_EN
        return templates.get(template_key, templates.get('no_results', ''))
    
    def _get_homeivf_link_action(self, language: str) -> SuggestedAction:
        """Get HomeIVF professional help link action (https://homeivf.com/)"""
        if language == 'hi':
            return SuggestedAction(
                type='link',
                label='पेशेवर सहायता – HomeIVF',
                url=settings.HOMEIVF_WEBSITE_URL,
                action='visit_homeivf'
            )
        else:
            return SuggestedAction(
                type='link',
                label='Get professional help (HomeIVF)',
                url=settings.HOMEIVF_WEBSITE_URL,
                action='visit_homeivf'
            )
    
    # Map topic/suggestion-chip phrases to FAQ category so the right content is returned
    TOPIC_PHRASE_TO_FAQ_CATEGORY: Dict[str, str] = {
        # English
        'lifestyle': 'lifestyle', 'lifestyle & diet': 'lifestyle', 'lifestyle and diet': 'lifestyle',
        'lifestyle and preparation': 'lifestyle', 'diet': 'lifestyle', 'preparation for ivf': 'lifestyle',
        'ivf diet': 'lifestyle', 'ivf lifestyle': 'lifestyle',
        'ivf process': 'process', 'ivf procedure': 'process', 'ivf procedures': 'process', 'process': 'process',
        'side effects': 'risks', 'medications': 'risks', 'medication': 'risks',
        'success rates': 'success_rates', 'costs': 'costs', 'costs & insurance': 'costs', 'costs and insurance': 'costs',
        'when to call doctor': 'support', 'general information': 'basics',
        # Hindi
        'जीवनशैली': 'lifestyle', 'जीवनशैली और आहार': 'lifestyle', 'आहार': 'lifestyle',
        'आईवीएफ प्रक्रिया': 'process', 'दुष्प्रभाव': 'risks', 'दवाएं': 'risks',
        'सफलता दर': 'success_rates', 'लागत': 'costs', 'लागत और बीमा': 'costs',
        'डॉक्टर को कब बुलाएं': 'support', 'सामान्य जानकारी': 'basics', 'तैयारी': 'lifestyle',
    }

    def _get_category_for_topic_query(self, message: str, language: str) -> Optional[str]:
        """
        If the user message is a known topic phrase (e.g. "Lifestyle & Diet", "जीवनशैली और आहार"),
        return the corresponding FAQ category so search returns the right topic.
        """
        if not message or not message.strip():
            return None
        msg = message.strip().lower()
        # Exact match first
        if msg in self.TOPIC_PHRASE_TO_FAQ_CATEGORY:
            return self.TOPIC_PHRASE_TO_FAQ_CATEGORY[msg]
        # Check if any topic phrase is contained (e.g. "Lifestyle & Diet" with extra spaces)
        for phrase, category in self.TOPIC_PHRASE_TO_FAQ_CATEGORY.items():
            if phrase in msg:
                return category
        return None

    def _is_follow_up_message(self, message: str) -> bool:
        """True if the message looks like a follow-up (short or continuation phrasing)."""
        if not message or not message.strip():
            return False
        msg = message.strip()
        word_count = len(msg.split())
        if word_count <= 4:
            return True
        follow_up_phrases = [
            r'what\s+about', r'how\s+about', r'and\s+the', r'and\s+cost', r'and\s+side\s+effect',
            r'tell\s+me\s+more', r'also', r'what\s+else', r'what\s+are\s+the',
            r'cost\s*\??', r'side\s+effect', r'risks?\s*\??', r'success\s+rate',
            r'how\s+long', r'when\s+should', r'क्या\s+और', r'और\s+क्या', r'बताएं', r'और'
        ]
        msg_lower = msg.lower()
        for pat in follow_up_phrases:
            if re.search(pat, msg_lower, re.IGNORECASE):
                return True
        return False

    def _get_previous_topic_from_history(self, history: List[Dict]) -> Optional[str]:
        """Return the last user message content from history (previous question/topic)."""
        if not history:
            return None
        for msg in reversed(history):
            if msg.get('sender') in ('patient', 'user'):
                content = (msg.get('content') or '').strip()
                if content:
                    return content
        return None

    def _get_contextualized_query(self, message: str, history: List[Dict], language: str) -> str:
        """
        Maintain chat context: for follow-up questions, combine with the previous
        user question so search and intent see the full topic (e.g. "what about side effects?"
        + previous "What is IVF?" -> "What is IVF what about side effects").
        """
        if not message or not message.strip():
            return message
        previous_topic = self._get_previous_topic_from_history(history)
        if not previous_topic or not self._is_follow_up_message(message):
            return message.strip()
        # Combine so search/intent get context (previous topic first for better semantic match)
        combined = f"{previous_topic} {message.strip()}"
        logger.debug(f"Contextualized query: {combined[:80]}...")
        return combined

    def _suggested_actions_from_faq_results(
        self,
        search_results: List[Dict],
        skip_index: int = 0,
        max_chips: int = 4,
    ) -> List[SuggestedAction]:
        """
        Build suggestion chips from FAQ/search results (question or title).
        Use these first so chips come from the knowledge base before template followups.
        """
        if not search_results or max_chips <= 0:
            return []
        seen = set()
        actions = []
        for i, result in enumerate(search_results):
            if i == skip_index:
                continue
            if len(actions) >= max_chips:
                break
            label = (result.get('question') or result.get('title') or '').strip()
            if not label or label.lower() in seen:
                continue
            seen.add(label.lower())
            actions.append(SuggestedAction(
                type='quick_reply',
                label=label,
                action='faq_followup',
                data={'question': label}
            ))
        return actions

    def _get_default_suggestions(self, language: str) -> List[SuggestedAction]:
        """Get default suggested actions (medication excluded - specialised opinion)"""
        if language == 'hi':
            return [
                SuggestedAction(
                    type='quick_reply',
                    label='आईवीएफ प्रक्रिया',
                    action='faq_process'
                ),
                SuggestedAction(
                    type='quick_reply',
                    label='अपॉइंटमेंट',
                    action='appointment_schedule'
                )
            ]
        else:
            return [
                SuggestedAction(
                    type='quick_reply',
                    label='IVF Process',
                    action='faq_process'
                ),
                SuggestedAction(
                    type='quick_reply',
                    label='Schedule Appointment',
                    action='appointment_schedule'
                )
            ]
    
    def _translate_action_label(self, label: str, language: str) -> str:
        """Translate action label to target language"""
        if language == 'hi':
            translations = {
                'Connect with Counsellor': 'काउंसलर से जुड़ें',
                'View Available Slots': 'उपलब्ध स्लॉट देखें',
                'Reschedule Appointment': 'अपॉइंटमेंट पुनर्निर्धारित करें'
            }
            return translations.get(label, translation_service.translate_to_hindi(label))
        return label
    
    def _filter_used_suggestions(
        self,
        suggested_actions: List[SuggestedAction],
        history: List[Dict],
    ) -> List[SuggestedAction]:
        """
        Remove suggestion chips (quick_reply) whose label or question
        was already sent by the user in this conversation.
        """
        if not history or not suggested_actions:
            return suggested_actions
        used_texts = set()
        for msg in history:
            if msg.get('sender') in ('patient', 'user'):
                content = (msg.get('content') or '').strip()
                if content:
                    used_texts.add(content.lower())
        if not used_texts:
            return suggested_actions
        filtered = []
        for action in suggested_actions:
            if action.type != 'quick_reply':
                filtered.append(action)
                continue
            # Compare label or stored question to what user already sent
            text = (action.label or '').strip() or (action.data or {}).get('question', '')
            text = (text or '').strip()
            if text and text.lower() in used_texts:
                continue  # skip already-used suggestion
            filtered.append(action)
        return filtered

    def _normalize_response_text(self, text: str) -> str:
        """
        Normalize AI-generated response text: fix broken list numbering,
        abbreviations (e.g. / i.e.), and excessive newlines.
        """
        if not text or not text.strip():
            return text
        # Fix "e. G." / "e. g." -> "e.g." (case-insensitive)
        text = re.sub(r'\be\.\s*g\.', 'e.g.', text, flags=re.IGNORECASE)
        # Fix "i. E." / "i. e." -> "i.e."
        text = re.sub(r'\bi\.\s*e\.', 'i.e.', text, flags=re.IGNORECASE)
        # Fix numbered list: "N.\\n\\nNext sentence" -> "N. Next sentence" so items read cleanly
        text = re.sub(r'(\d+)\.\s*\n\s*\n\s*([A-Za-z])', r'\1. \2', text)
        # Collapse 3+ newlines to double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _response_language(self, answer_text: str, requested_language: str) -> str:
        """
        Use answer language for disclaimer and suggestions so the full response is
        monolingual when possible. If the answer is in Hindi (Devanagari), use Hindi
        for disclaimer and suggestion labels.
        """
        if not answer_text:
            return requested_language
        # Devanagari script range
        if re.search(r'[\u0900-\u097F]', answer_text):
            return 'hi'
        return requested_language

    def _get_urgent_message(self, language: str) -> str:
        """Get urgent escalation message"""
        if language == 'hi':
            return "यदि यह एक चिकित्सा आपातकाल है, तो कृपया तुरंत [आपातकालीन नंबर] पर कॉल करें।"
        return "If this is a medical emergency, please call [Emergency Number] immediately."
    
    def _infer_category_from_query(self, query: str, intent: str) -> str:
        """
        Infer category from query and intent for generating follow-ups and suggestions
        
        Args:
            query: User query
            intent: Detected intent
        
        Returns:
            Category string (e.g., 'ivf_process', 'medication', 'side_effects', etc.)
        """
        query_lower = query.lower()
        
        # First, try to extract from intent
        if intent and intent.startswith('faq_'):
            category_key = intent.replace('faq_', '')
            category_mapping = {
                'process': 'ivf_process',
                'medication': 'medication',
                'side_effects': 'side_effects',
                'lifestyle': 'lifestyle',
                'success_factors': 'success_factors',
                'costs': 'costs',
                'general': 'general'
            }
            if category_key in category_mapping:
                return category_mapping[category_key]
        
        # If intent doesn't help, infer from query keywords
        # Keywords for different categories
        category_keywords = {
            'ivf_process': ['process', 'procedure', 'step', 'cycle', 'how long', 'duration', 'timeline', 
                           'प्रक्रिया', 'चक्र', 'समय', 'कितना समय'],
            'medication': ['medication', 'medicine', 'drug', 'injection', 'dose', 'prescription',
                          'दवा', 'दवाएं', 'इंजेक्शन', 'खुराक'],
            'side_effects': ['side effect', 'adverse', 'problem', 'issue', 'pain', 'discomfort', 'symptom',
                            'दुष्प्रभाव', 'समस्या', 'दर्द', 'लक्षण'],
            'lifestyle': ['diet', 'food', 'exercise', 'activity', 'lifestyle', 'restriction',
                        'आहार', 'खाना', 'व्यायाम', 'जीवनशैली'],
            'success_factors': ['success', 'rate', 'chance', 'probability', 'factor', 'outcome',
                              'सफलता', 'दर', 'संभावना', 'कारक'],
            'costs': ['cost', 'price', 'fee', 'insurance', 'payment', 'afford',
                     'लागत', 'मूल्य', 'बीमा', 'भुगतान']
        }
        
        # Count matches for each category
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score, or 'general' if no matches
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        return 'general'
    
    def _handle_medgemma_fallback(
        self,
        message: str,
        intent: str,
        language: str,
        conversation_context: Dict,
        search_results: List[Dict],
        max_followups: int = 3,
        contextualized_query: Optional[str] = None,
    ) -> Tuple[str, List[SuggestedAction], List[RelatedContent]]:
        """
        Handle Medgemma fallback logic (consolidated to avoid code duplication).
        Uses contextualized_query when provided to maintain chat context.
        
        Returns:
            Tuple of (response_text, suggested_actions, related_content)
        """
        # Build context from conversation history (optimized)
        context_text = None
        if conversation_context.get('history'):
            recent_history = conversation_context['history'][-3:]
            context_parts = [
                f"{msg.get('sender', 'user')}: {msg.get('content', '')}"
                for msg in recent_history
            ]
            context_text = ' '.join(context_parts)
        
        suggested_actions = []
        related_content = []
        response_text = None
        query_for_medgemma = (contextualized_query or message).strip()
        
        # Try Medgemma if enabled (with IVF guardrail)
        if settings.USE_MEDGEMMA:
            try:
                # Enhance query with IVF context for better results (use contextualized query when available)
                enhanced_query = self.ivf_guardrail.enhance_ivf_context(query_for_medgemma, language)
                image_b64 = conversation_context.get('image_base64')
                medgemma_answer = self.knowledge_engine.get_answer_from_medgemma(
                    enhanced_query,
                    language=language,
                    context=context_text,
                    image=image_b64,
                )
                
                # Validate that response is IVF-related
                if medgemma_answer:
                    is_valid, rejection_msg = self.ivf_guardrail.validate_response(
                        medgemma_answer, message, language
                    )
                    if not is_valid:
                        logger.warning("Medgemma response failed IVF validation")
                        medgemma_answer = None
                
                if medgemma_answer:
                    logger.debug(f"Medgemma answer generated ({len(medgemma_answer)} chars)")
                    response_text = medgemma_answer
                    # Use answer language for disclaimer and suggestions so response is monolingual
                    response_lang = self._response_language(medgemma_answer, language)
                    
                    # Add note that this is from AI model (same language as answer)
                    ai_note = (
                        "\n\n(यह जानकारी AI मॉडल से प्राप्त की गई है। कृपया अपने डॉक्टर से भी परामर्श करें।)"
                        if response_lang == 'hi'
                        else "\n\n(This information is generated by an AI model. Please also consult with your doctor.)"
                    )
                    response_text += ai_note
                    
                    # Extract category from query/intent for generating follow-ups and suggestions
                    inferred_category = self._infer_category_from_query(message, intent)
                    
                    # First: suggestion chips from FAQs when we have search results
                    if search_results:
                        faq_chips = self._suggested_actions_from_faq_results(
                            search_results, skip_index=0, max_chips=max_followups
                        )
                        suggested_actions.extend(faq_chips)
                        faq_labels = [a.label for a in faq_chips]
                    else:
                        faq_labels = []
                    # Then: template follow-up questions, excluding FAQ chips already added
                    if settings.ENABLE_FOLLOWUPS:
                        followups = self.followup_generator.generate_followups(
                            category=inferred_category,
                            language=response_lang,
                            answer=medgemma_answer,
                            exclude_questions=faq_labels if faq_labels else None
                        )
                        for followup in followups[:max_followups]:
                            suggested_actions.append(SuggestedAction(
                                type='quick_reply',
                                label=followup,
                                action='faq_followup',
                                data={'question': followup}
                            ))
                    
                    # Add proactive suggestions (same language as answer)
                    if settings.ENABLE_PROACTIVE_SUGGESTIONS:
                        proactive = self.proactive_suggestions.get_suggestions(
                            current_topic=inferred_category,
                            language=response_lang,
                            conversation_history=conversation_context.get('history', [])
                        )
                        suggested_actions.extend(proactive[:max_followups])
                    
                    # Add related content from knowledge base search
                    if search_results:
                        for result in search_results[:max_followups]:
                            related_content.append(RelatedContent(
                                title=result.get('title', result.get('question', '')),
                                id=result.get('article_id', result.get('faq_id', '')),
                                type=result.get('type', 'faq')
                            ))
                else:
                    logger.warning("Medgemma returned no answer")
            except Exception as e:
                logger.error(f"Medgemma error: {e}", exc_info=True)
        else:
            logger.debug("Medgemma disabled")
        
        # If medgemma fails, return no results
        if not response_text:
            logger.warning(f"No answer found for: {message[:50]}")
            response_text = self._get_template('no_results', language)
            suggested_actions.append(SuggestedAction(
                type='button',
                label=self._translate_action_label('Connect with Counsellor', language),
                action='escalate'
            ))
        
        return response_text, suggested_actions, related_content