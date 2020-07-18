import json


def getJsonData(jsonFile):
	f = open(jsonFile)
	data = json.load(f)
	f.close()
	return data