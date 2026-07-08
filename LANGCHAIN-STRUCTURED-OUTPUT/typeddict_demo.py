from typing import TypedDict

class User(TypedDict):

    name : str
    age : int

new_person: User = {'name': 'sonali', 'age': 25}

print(new_person)