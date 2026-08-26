import asyncio
from sqlalchemy import select, func
from app.db.session import get_db_session
from app.db.models import ResearchRun, SourceRow


async def main():
    async with get_db_session() as session:
        runs = (
            await session.execute(
                select(ResearchRun).order_by(ResearchRun.created_at.desc())
            )
        ).scalars().all()
        print(f"{len(runs)} runs in DB:")
        for r in runs:
            counts = (
                await session.execute(
                    select(SourceRow.status, func.count())
                    .where(SourceRow.run_id == r.run_id)
                    .group_by(SourceRow.status)
                )
            ).all()
            print(
                f"  run_id={r.run_id}  topic={r.topic[:50]!r}  "
                f"created={r.created_at}  source_status_counts={dict(counts)}"
            )


if __name__ == "__main__":
    asyncio.run(main())