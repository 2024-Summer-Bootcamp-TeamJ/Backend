import sys, os

sys.path.append("app/")
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from database import Base, engine
from models import *

Base.metadata.create_all(bind=engine)
