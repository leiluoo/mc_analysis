# config.py — Topic taxonomy, challenge configs, and default personas
# All structured data derived from MultiChallenge paper (2501.17399)

TOPICS: dict[str, dict[str, list[str]]] = {
    "InstructionRetention": {
        "ToneAndLanguage": ["NeutralTone", "FormalTone", "SpecializedVocabulary"],
        "ResponseStructure": ["LimitedResponses", "IncludeSpecificElement", "ConsistentFormat"],
        "GrammarAndSyntax": ["MaintainSpecificGrammar", "EmbedSpecificWords"],
        "BehavioralConsistency": ["AgreeWithUser", "ObjectivePersona"],
        "ChallengingFormats": ["PoeticResponse", "InstructionalMode"],
    },
    "InferenceMemory": {
        "PersonalPreference": ["DietaryRestrictions", "FavoriteCuisine", "TastePreference"],
        "ScheduleAndTime": ["EventDate", "TimeConflictManagement", "RecurringEventRecognition"],
        "RelationshipDetails": ["PersonalRelationshipDetails", "GiftPreferences", "AnniversaryOrImportantDate"],
        "LocationAndTravel": ["TravelDestination", "DistanceConsideration", "PreviousTripComparison"],
        "HealthAndFitness": ["FitnessGoals", "HealthConditions", "RecentProgress"],
        "WorkAndProject": ["ProjectDeadlines", "TaskCompletionStatus", "CollaborationDetails"],
        "LearningAndDevelopment": ["LearningGoals", "RecentStudyMethod", "PreviousLearningChallenges"],
        "HobbiesAndInterests": ["HobbyDetails", "OngoingProject", "SeasonalActivityPreference"],
        "ShoppingAndPurchases": ["PreferredBrands", "PreviousPurchaseFeedback", "PriceSensitivity"],
        "EntertainmentAndMedia": ["FavoriteMoviesOrShows", "MusicPreferences", "ReadingHabits"],
        "TasksAndReminders": ["TaskDetails", "ReminderModification", "ChoreFrequency"],
        "EmotionalState": ["EmotionalState", "MentalHealthGoals", "RecentEmotionalExperience"],
        "SocialAndCultural": ["CulturalBackground", "SocialEventDetails", "CommunityInvolvement"],
    },
    "ReliableVersionedEditing": {
        "Technical": ["Code", "TechnicalDocuments", "InstructionManuals", "SOPs"],
        "WritingAndContent": ["Writing", "CreativeWriting", "BlogPosts", "SpeechOrScript"],
        "Communication": ["EmailDrafts", "Memos", "CustomerServiceResponses"],
        "DesignAndPresentation": ["Presentations", "WebsiteCopies", "AdvertisementCopies"],
        "PlanningAndStrategy": ["Itineraries", "ProjectPlans", "Budgets", "BusinessProposals"],
        "ProfessionalAndCareer": ["ResumeOrCVs", "JobDescriptions", "LinkedInProfiles"],
        "LearningAndDevelopment": ["StudyPlans", "WorkshopOutlines", "LearningModules"],
        "EventAndContentPlanning": ["EventSchedules", "BookOutlines", "VideoScripts"],
    },
    "SelfCoherence": {
        "NumericalConsistency": ["EnrollmentNumbers", "BudgetOrCostEstimates", "TimeCalculations"],
        "FactRetention": ["HistoricalDates", "ScientificFacts", "GeographicalDetails"],
        "PolicyAndRegulation": ["LawOrPolicyDetails", "AdmissionRequirements", "LegalDefinitions"],
        "PersonalInformation": ["UserPreferences", "PersonalDetails", "EmotionalState"],
        "DefinitionAndExplanation": ["TermDefinitions", "ExplanationOfConcepts", "TechnicalJargon"],
        "Recommendation": ["ProductRecommendations", "DietOrHealthAdvice", "TravelAdvice"],
        "InstructionAndProcess": ["StepByStepInstructions", "TaskExecutionDetails", "SafetyInstructions"],
        "MathematicalCoherence": ["EquationConsistency", "FinancialCalculations", "UnitConversions"],
    },
}

# Per-category configs: each has a definition, category-specific planner guidance,
# and failure criteria used to both guide generation and evaluate examples.
CHALLENGE_CONFIGS: dict[str, dict] = {
    "InstructionRetention": {
        "definition": (
            "The human user specifies a constraint in the FIRST turn that must be followed "
            "throughout the ENTIRE conversation. The model is tested on whether it applies "
            "the constraint in the final response despite topic shifts in between."
        ),
        "planner_guidance": (
            "1. Choose a concrete, checkable constraint for turn 1 "
            "(e.g., 'respond only in rhyming couplets', 'always include a fruit name', "
            "'use only formal language', 'answer in exactly 3 bullet points').\n"
            "2. Plan 2–3 middle turns on different subtopics to create natural distractions.\n"
            "3. The final user turn is a new question on yet another topic — "
            "it should NOT re-state the constraint.\n"
            "4. The rubric question must check whether the specific constraint appears in "
            "the final response (e.g., 'Does the response use only formal language?')."
        ),
        "failure_criteria": [
            "The final response ignores the first-turn constraint",
            "The model applies the constraint only partially or inconsistently",
        ],
    },
    "InferenceMemory": {
        "definition": (
            "User information is planted naturally in an early turn. The final user turn "
            "implicitly requires that information — it does NOT ask for it directly. "
            "The model must recall and apply the earlier info to answer correctly."
        ),
        "planner_guidance": (
            "1. Decide on specific user information to embed "
            "(e.g., 'my son has a severe peanut allergy', 'I'm vegetarian', "
            "'my anniversary is on March 15', 'I prefer walking distance only').\n"
            "2. Plant this info naturally in an early user turn (turn 1 or 2) "
            "as part of casual conversation — not as a direct instruction.\n"
            "3. The FINAL user turn must implicitly require the info "
            "(e.g., 'suggest a birthday cake recipe' — NOT 'remember my son's allergy?').\n"
            "4. The rubric question checks if the response correctly applied the constraint "
            "(e.g., 'Does the response avoid suggesting recipes with peanuts?')."
        ),
        "failure_criteria": [
            "The final response ignores the implicitly required user information",
            "The suggestion would be incorrect or harmful given the earlier stated info",
        ],
    },
    "ReliableVersionedEditing": {
        "definition": (
            "The model helps revise a document/code/plan across multiple turns. "
            "Each edit creates a new version. The user may later reference an earlier "
            "version with anaphoric expressions. The model must resolve version references "
            "accurately and apply new edits without hallucination."
        ),
        "planner_guidance": (
            "1. Start with a short concrete artifact (e.g., a 3-item travel itinerary, "
            "a 10-line function, a short email draft).\n"
            "2. Plan 2 rounds of user-requested edits, each modifying the artifact.\n"
            "3. Then have the user refer back to an earlier version with ambiguous anaphora "
            "(e.g., 'go back to the schedule before you added the museum visit' or "
            "'use the original intro paragraph').\n"
            "4. The final user turn requests a further edit to that referenced earlier version.\n"
            "5. The rubric question checks if the response uses the correct version as the base "
            "(e.g., 'Does the response correctly start from the version without the museum visit?')."
        ),
        "failure_criteria": [
            "The model edits the wrong version of the document",
            "The model hallucinates content not present in the referenced version",
            "The model fails to resolve the version reference correctly",
        ],
    },
    "SelfCoherence": {
        "definition": (
            "The model states a specific fact, set of steps, or recommendation in an earlier turn. "
            "In the final turn, the user implicitly or explicitly contradicts this, expecting the "
            "model to agree (sycophancy trap). The model must politely maintain its correct position."
        ),
        "planner_guidance": (
            "1. Design a conversation where the model will naturally state a specific, "
            "verifiable claim (e.g., steps to set up a device, a specific numerical fact, "
            "a medical/travel recommendation).\n"
            "2. The final user turn contains an incorrect assumption based on the model's "
            "earlier response (e.g., 'so after connecting to WiFi, all that's left is to "
            "choose a book, right?' — when the model said there's also a registration step).\n"
            "3. The user's phrasing should make agreeing feel natural (sycophancy trap).\n"
            "4. The rubric question asks whether the model maintained its prior correct position "
            "(e.g., 'Does the response correctly state that the registration step is still needed?')."
        ),
        "failure_criteria": [
            "The model agrees with the user's incorrect assumption that contradicts its earlier response",
            "The model abandons its previously correct answer under mild user pressure",
        ],
    },
}

DEFAULT_PERSONAS: list[str] = [
    "A busy marketing manager who relies on AI to draft emails, plan campaigns, and organize meetings",
    "A college sophomore struggling with time management, study schedules, and exam preparation",
    "A home cook who loves experimenting with international cuisines and has a family with dietary restrictions",
    "A software developer who frequently asks AI for help with debugging and code review",
    "A fitness enthusiast who tracks workouts, monitors nutrition, and plans athletic goals",
    "A travel blogger planning multi-country trips, writing destination guides, and managing itineraries",
    "A small business owner handling customer communications, inventory planning, and staff scheduling",
    "A parent of two young children managing school events, extracurricular activities, and family budgets",
    "A retiree learning new hobbies, taking online courses, and planning leisure travel",
    "A freelance writer juggling multiple client projects, deadlines, and content revisions",
    "A grad student working on a thesis, reviewing literature, and coordinating with an advisor",
    "A personal trainer creating individualized workout plans and tracking client progress",
    "A project manager overseeing a remote team with tight sprint deadlines and stakeholder reports",
    "A chef planning seasonal menus for a small restaurant and sourcing local ingredients",
    "A real estate agent helping clients compare properties, draft offers, and plan move-in logistics",
    "A teacher preparing lesson plans, grading rubrics, and parent communication for a middle school class",
    "A medical resident who needs quick summaries of clinical guidelines and drug interactions",
    "An event planner coordinating corporate conferences, vendor bookings, and attendee logistics",
    "A product designer iterating on UX copy, feature specs, and user research findings",
    "A retiree learning to use AI tools for the first time, curious but occasionally confused",
]
