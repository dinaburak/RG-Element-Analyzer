import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from sqlmodel import SQLModel, Field, create_engine, Session, select
from typing import Optional

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)

app = FastAPI()


class RGElement(SQLModel, table=True):
    __tablename__ = "rg_elements"

    id: Optional[int] = Field(default=None, primary_key=True)
    element_name: str
    body_difficulty_category: str
    base_value: float
    knee_angle: float | None = None
    hip_angle: float | None = None
    knee_tolerance: float | None = None
    hip_tolerance: float | None = None

@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)


@app.get("/")
def home():
    return {"message": "Rhythmic Gymnastics Judge API is working"}

@app.get("/elements/search")
def get_elements(DBname: str):
    with Session(engine) as session:
        statement = select(RGElement).where(
            RGElement.element_name.ilike(f"%{DBname}%")
         )
        return session.exec(statement).all()
    
@app.get("/elements")
def get_elements(DBcategory: Optional[str] = None):
    with Session(engine) as session:
        statement = select(RGElement)

        if DBcategory:
            statement = statement.where(
                RGElement.body_difficulty_category == DBcategory
            )

        return session.exec(statement).all()

@app.post("/elements")
def create_element(element: RGElement):
    with Session(engine) as session:
        session.add(element)
        session.commit()
        session.refresh(element)
        return element
    
