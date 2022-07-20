import streamlit


@streamlit.experimental_memo
def secrets():
    #     consumer_key = 'HL9UWl40pWj62f0msQ6Fdx2ZF'
    #     consumer_secret = 'JdKmzL4TjLEpXuBstAWES3F9Bn49oyxTwsy34ojPoSQr3I7qet'
    consumer_key = "DEnyjO0x2iXiZOSliyqTz5KPY"
    consumer_secret = "qsyEjJbmsWVMqMQegHZ6DSs17jYRgtwFaRMnQVpLY7u9UL4B9x"
    return consumer_key, consumer_secret


def access():
    #     access_token = '483078594-Qw2r1HpBeb7YQDzVbXINMFrtwb7g8mSxJb4bvDaH'
    #     access_token_secret ='5afGvv9FQL6Je6Wou9PhnGYQzVWrWL6w4dLt6mKa9xXy9'
    access_token = "1284030681895579648-8SZpcRa7MWdmjvoFcL3C5KEDVwRAPJ"
    access_token_secret = "ylYrrw6UX9dXE89Lixwo4vFaZdNOGQqcPCGS5EHH3lWg5"
    return access_token, access_token_secret
