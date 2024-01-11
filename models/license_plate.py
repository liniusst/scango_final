from database.base import Base
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship


class LicensePlate(Base):
    __tablename__ = "license_plates"

    id = Column(Integer, primary_key=True, index=True)
    owner = Column(String)
    license_plate = Column(String, index=True)

    entries = relationship("Entries", back_populates="license_plate")

    def __init__(self, owner, license_plate):
        self.owner = owner
        self.license_plate = license_plate
