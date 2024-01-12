from database.base import Base, Session, engine
from models.entries import Entries
from models.license_plate import LicensePlate


Base.metadata.create_all(engine)
session = Session()


def get_all_plates():
    all_plates = session.query(LicensePlate).all()
    return all_plates


def create_plates(owner: str, plate: str):
    data = LicensePlate(owner, plate)
    session.add(data)
    session.commit()
    session.close()
