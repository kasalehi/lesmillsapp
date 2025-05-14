import streamlit_authenticator as stauth

# List of plain passwords
passwords = ['lesmills','lesmills']

# Proper hashing for multiple passwords (as a list)
hashed_passwords = stauth.Hasher(passwords).generate()

# Print output to paste in your app
print(hashed_passwords)

