# Allows to generate colour
from faker import Factory
fake = Factory.create()
color = fake.hex_color()
