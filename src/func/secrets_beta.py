import streamlit
@streamlit.experimental_memo
def secrets():
    consumer_key = 'HL9UWl40pWj62f0msQ6Fdx2ZF'
    consumer_secret = 'JdKmzL4TjLEpXuBstAWES3F9Bn49oyxTwsy34ojPoSQr3I7qet'
    return consumer_key, consumer_secret

def access():
    access_token = '483078594-Qw2r1HpBeb7YQDzVbXINMFrtwb7g8mSxJb4bvDaH'
    access_token_secret ='5afGvv9FQL6Je6Wou9PhnGYQzVWrWL6w4dLt6mKa9xXy9'
    return  access_token, access_token_secret