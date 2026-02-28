import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), "ChefsHatGYM", "src"))

from rooms.room import Room

env = Room(
    run_remote_room=False,
    room_name="test_room",
    max_matches=3
)

print("Room created OK")
print("Type:", type(env))