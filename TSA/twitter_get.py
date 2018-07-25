#!usr/bin/env python

from twitter import *

t = Twitter(auth=OAuth(
				consumer_key='J5i4LfYBIaC7tr5yaL51C5t0A',
				consumer_secret='tsDpTAECaJ17Itw0RXowdBS3Tr3i9MGcLTgI0tvg8I96BvpSWS',
				token='230478664-OuxQhjxqfzryAfjc19Y0OXb5qOGCmQa7AQa3Fapv',
				token_secret='L081yUmohtGMd3j3taPxd1E1lV9FugCgogPcBk1ZkrujH'))



print(t.search.tweets(q="@LaCasaInvisible", tweet_mode='extended'))