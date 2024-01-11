from datetime import datetime
from models.entries import Entries
from models.license_plate import LicensePlate
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import NoResultFound


def create_entrie(db: Session, license_plate_id: int, license_plate: str) -> Entries:
    try:
        entrie = (
            db.query(LicensePlate).filter(LicensePlate.id == license_plate_id).first()
        )
        now = datetime.now()
        if entrie:
            new_entrie = Entries(
                license_plate_id=new_entrie.id,
                license_plate=license_plate,
                entrie_time=now,
            )
            db.add(new_entrie)
            db.commit()
            db.refresh(new_entrie)
            return new_entrie
        else:
            raise NoResultFound("License plate not found for entries creation")
    except TypeError as error:
        raise TypeError("License plate not found for entrie creation")
    except Exception as error:
        db.rollback()
        return None


def get_entries_by_license_plate_id(
    db: Session, license_plate_id: int
) -> list[Entries]:
    try:
        license_plate = (
            db.query(LicensePlate).filter(LicensePlate.id == license_plate_id).first()
        )
        if license_plate:
            return license_plate.entries
        else:
            raise NoResultFound("License plate not found for entries retrieval")
    except Exception as error:
        return []
