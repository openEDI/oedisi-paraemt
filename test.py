from pydantic import BaseModel


class SubTest(BaseModel):
    name: str
    age: int


class Test(BaseModel):
    name: str = "John Doe"
    is_student: bool
    age: int
    friend: SubTest


json_example = """
{
    "age": 25,
    "is_student": false,
    "friend": {
        "name": "Jane Doe",
        "age": 24
    }
}
"""

test = Test.parse_raw(json_example)
print(test.name)
print(test.age)
print(test.is_student)
print(test.friend)

print(test.json(indent=2))
