from database.base import Base
from sqlalchemy import Column, DateTime, ForeignKey, Integer
from sqlalchemy.orm import relationship


class Entries(Base):
    __tablename__ = "entries"

    id = Column(Integer, primary_key=True, index=True)
    license_plate_id = Column(Integer, ForeignKey("license_plates.id"))
    entrie_time = Column(DateTime)

    license_plate = relationship("LicensePlate", back_populates="entries")

    def __init__(self, license_plate_id, entrie_time):
        self.license_plate_id = license_plate_id
        self.entrie_time = entrie_time
