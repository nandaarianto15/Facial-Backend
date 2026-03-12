# database.py
import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime

# Mengambil URL dari Environment Variable. 
# Jika tidak ada (misal di lokal), pakai default.
DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:@localhost/mirrasense")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- MODEL: USER ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(255))
    email = Column(String(255))
    tel = Column(String(255))
    created_at = Column(DateTime, default=datetime.now)

# --- MODEL: BLOG ---
class Blog(Base):
    __tablename__ = "blog"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    title = Column(String(255))
    slug = Column(String(255), unique=True, index=True)
    content = Column(Text)
    category = Column(String(100))
    tag = Column(String(255))
    image = Column(String(255))
    published_at = Column(DateTime, default=datetime.now)
    is_published = Column(Boolean, default=True)
    author_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.now)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()