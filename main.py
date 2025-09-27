from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
load_dotenv()

def main():
    information = """
African Head Charge is a psychedelic dub ensemble active since 1981, when they released their debut album, recorded at Berry Street Studio in London, which was, at the time, run by Dennis Bovell.[1] The group was formed by percussionist Bonjo Iyabinghi Noah,[2] and featured a revolving cast of members, including the original members of Creation Rebel, Undivided Roots, Carlton "Bubblers" Ogilvie and Crucial Tony Phillps of Ruff Cutt, Style Scott of The Roots Radics and The Dub Syndicate, George Oban, (bassist who had played with Aswad and Burning Spear), Headley Bennett, Prisoner, Crocodile, (both pseudonyms for Adrian Sherwood), Nick Plytas (originally from 1970s pub-rock proto punk band Roogalator), Junior Moses, Sunny Akpan of The Funkees, Steve Beresford of Brian Eno's Portsmouth Sinfonia, Bruce Smith of Public Image Ltd, Evar Wellington of British Roots Reggae band, The Makka Bees, Skip McDonald, Gaudi and Jah Wobble. Martin Frederix, sound engineer and live-mixer for This Heat also contributed to the band, playing bass and mixing some of the tracks on Songs of Praise. The group released most of its albums on Adrian Sherwood's label, On-U Sound, with much of the iconic sleeve-design artwork provided by noted photographer, Kishi Yamamoto, who also played keyboards, Guzheng Chinese Harp and Pipa Chinese lute on some of the compositions.[1]

Journalist David Stubbs, writing in The Wire, said, "The notion of African Head Charge was hatched when Adrian Sherwood read Brian Eno's comment about his vision for a 'psychedelic Africa'".[3]

All of their early work has been re-released as double albums on vinyl and CD.
"""
    summary_template = """
    given the information {information} about a band I want you to create:
    1. A short summary.
    2. two interesting facts about the band.
    3. A list of three albums by the band."""

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-5")
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
