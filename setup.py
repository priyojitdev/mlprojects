from setuptools import find_packages,setup
from typing import List
HYPHEN_E_DOT="-e ."

def get_req(file_path: str) -> List[str]:
    req=[]
    with open(file_path) as f:
        req=f.readlines()
        req=[r.replace("\n","") for r in req]
        if HYPHEN_E_DOT in req:
            req.remove(HYPHEN_E_DOT)


    return req

setup(
    name="mlproject",
    version="0.0.1",
    author="Priyojit",
    author_email="priyojitmaji2745@gmail.com",
    packages= find_packages(),
    install_requires=get_req("requirements.txt"), 
    
)