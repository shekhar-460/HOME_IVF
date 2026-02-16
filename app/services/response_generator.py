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
        # Check full response cache first (for exact message matches)
        cache_key = f"response:{hashlib.md5(f'{message}:{language}'.encode()).hexdigest()}"
        if hasattr(self.knowledge_engine, 'redis_client') and self.knowledge_engine.redis_client:
            try:
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
        intent_task = self.intent_classifier.classify(message, language, conversation_context)
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
            # PRIMARY: Search knowledge base (JSON file first, then database)
            search_results = self.knowledge_engine.search(
                message,
                language=language,
                top_k=settings.TOP_K_SEARCH_RESULTS
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
                
                # Generate follow-up questions (same language as answer)
                if settings.ENABLE_FOLLOWUPS:
                    followups = self.followup_generator.generate_followups(
                        category=category,
                        language=response_lang,
                        answer=response_text
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
                # FALLBACK: Use medgemma if FAQ doesn't have good match
                response_text, suggested_actions, related_content = self._handle_medgemma_fallback(
                    message, intent, language, conversation_context, search_results
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
            # Generic response - try FAQ first, then medgemma
            search_results = self.knowledge_engine.search(message, language, top_k=1)
            
            if search_results and search_results[0]['similarity'] >= settings.MIN_CONFIDENCE_SCORE:
                response_text = search_results[0].get('answer', search_results[0].get('content', ''))
                category = search_results[0].get('category') or 'general'
                response_lang = self._response_language(response_text, language)
                
                # Add follow-ups and suggestions (same language as answer)
                if settings.ENABLE_FOLLOWUPS:
                    followups = self.followup_generator.generate_followups(
                        category=category,
                        language=response_lang,
                        answer=response_text
                    )
                    for followup in followups[:2]:  # Limit to 2 for generic responses
                        suggested_actions.append(SuggestedAction(
                            type='quick_reply',
                            label=followup,
                            action='faq_followup'
                        ))
            else:
                # Try medgemma as fallback (reuse common method)
                response_text, new_actions, new_content = self._handle_medgemma_fallback(
                    message, intent, language, conversation_context, search_results, max_followups=2
                )
                suggested_actions.extend(new_actions)
                related_content.extend(new_content)
                
                if not response_text:
                    if confidence < 0.6:
                        response_text = self._get_template('low_confidence', language)
                    else:
                        response_text = self._get_template('no_results', language)
        
        # Always add HomeIVF link to Actions section (use answer language when answer is in Hindi)
        effective_language = self._response_language(response_text or '', language)
        homeivf_action = self._get_homeivf_link_action(effective_language)
        # Check if HomeIVF link is not already in suggested_actions
        if not any(action.url == settings.HOMEIVF_WEBSITE_URL for action in suggested_actions):
            suggested_actions.append(homeivf_action)
        
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
        # Only cache FAQ-based responses (not Medgemma) for faster retrieval
        if hasattr(self.knowledge_engine, 'redis_client') and self.knowledge_engine.redis_client:
            try:
                # Check if this was a FAQ response (high confidence, not from Medgemma)
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
        max_followups: int = 3
    ) -> Tuple[str, List[SuggestedAction], List[RelatedContent]]:
        """
        Handle Medgemma fallback logic (consolidated to avoid code duplication)
        
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
        
        # Try Medgemma if enabled (with IVF guardrail)
        if settings.USE_MEDGEMMA:
            try:
                # Enhance query with IVF context for better results
                enhanced_query = self.ivf_guardrail.enhance_ivf_context(message, language)
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
                    
                    # Generate follow-up questions (same language as answer)
                    if settings.ENABLE_FOLLOWUPS:
                        followups = self.followup_generator.generate_followups(
                            category=inferred_category,
                            language=response_lang,
                            answer=medgemma_answer
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