from app.services.agent import AgentConfig, LessonPlanningAgent


def generate_lesson_from_query(
    user_prompt: str,
    subject: str | None = None,
    grade_level: str | None = None,
    curriculum: str | None = None,
    topic: str | None = None,
    retrieval_limit: int = 5,
) -> dict:
    """Legacy wrapper: generate a lesson script via the agent pipeline."""
    agent = LessonPlanningAgent(config=AgentConfig(retrieval_limit=retrieval_limit))
    return agent.run(
        user_prompt=user_prompt,
        subject=subject,
        grade_level=grade_level,
        curriculum=curriculum,
        topic=topic,
        retrieval_limit=retrieval_limit,
    )
