from database.queries import create_meeting

meeting_id = create_meeting(
    title="Database Test Meeting",
    department_id="b6f8468a-477c-4045-a696-c402afae99a5"
)

print("Meeting Created")
print(meeting_id)