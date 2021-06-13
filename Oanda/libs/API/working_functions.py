

import json

def AccountID(self, tags=[]):
    'Gives all accountID of the access_token with specified tags'
    accountID = []
    for i in self.AccountList()["accounts"]:
        if i["tags"] == tags:
            accountID.append(i["id"])
    return accountID

def readable_output(output, sort_keys=True, indent = 4, separators=(',', ': ')):
    return json.dumps(output, sort_keys=sort_keys, indent=indent, separators=separators)

def CallableFunctions(Class):
    return [print(func) for func in dir(Class) if callable(getattr(Class, func, __doc__))]

# def DictionaryCheck():
#     print(ReadableOutput(check.AccountList()))
    
#     data_alt = {} 
#     data_alt['accounts'] = []
#     data_alt['accounts'].append({})
#     data_alt['accounts'][0]['id'] = '101-004-16661696-002'
#     data_alt['accounts'][0]['tags'] = []
#     data_alt['accounts'].append({})
#     data_alt['accounts'][1]['id'] = '101-004-16661696-001'
#     data_alt['accounts'][1]['tags'] = []
    
#     print(ReadableOutput(data_alt))