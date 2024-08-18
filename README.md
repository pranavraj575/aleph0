# AlephZero: Extending AlphaZero to Infinite Boards

General implementation of $\aleph_0$,
an extension of AlphaZero which uses Transformer network architecture to solve a class of games 
($\aleph_0$-extensive form games) whose observation space is sequential, 
and whose action space arises from choosing indices in this observation space.

## Installation
* Either from pip: ```pip3 install aleph0```
* Or clone from repo:
  ```bash
  git clone https://github.com/pranavraj575/aleph0
  pip3 install -e aleph0
  ```
To play Jenga, must install additional packages: ```pip3 install aleph0[jenga]```
