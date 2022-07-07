from databases import Database
import pickle
import asyncio
class ResultLogger:
    
    def __init__(self, folder="logs", experiment="test"):
        self.folder=  folder
        self.experiment = experiment
    
#     @asyncio.coroutine
    async def connect(self):    
        self.database = Database('sqlite+aiosqlite:///%s/%s.db'%(self.folder, self.experiment))
#         self.database
        return self.database
        
#     @asyncio.coroutine
    async def makeSchema(self):       
        query = """CREATE TABLE IF NOT EXISTS results (   settings VARCHAR(1024) PRIMARY KEY, resultdata LONGBLOB)"""
        await self.database.execute(query=query)

        
    def makeConnection(self):
        asyncio.run( self.connect() )
        asyncio.run( self.makeSchema() )
           
#     @asyncio.coroutine        
    async def checkIfExistsCoroutine(self, _setting):    
        return await self.database.fetch_one(query="SELECT * FROM results where settings=:sett",values={"sett":_setting})
    
    def checkIfExists(self, _setting):    
        return asyncio.run( self.checkIfExistsCoroutine(_setting) )       
        
#     @asyncio.coroutine        
    def lockSetting(self, _setting):    
        try:
            asyncio.run(self.database.execute_many(query="insert into results (settings)  values ( :sett) ", values=[{"sett":_setting}]))
        except:
            pass
    
    
    def updateSetting(self, _setting, data):    
        asyncio.run(self.database.execute_many(query="update results set resultdata = :data where settings =  :sett ", 
                                values=[{
                                    "data" : pickle.dumps(data),
                                    "sett":_setting}])
                   )
    
#     @asyncio.coroutine    
    def getData(self, _setting):    
        rows = asyncio.run( self.database.fetch_one(query="SELECT * FROM results where settings=:sett",values={"sett":_setting}) )
        return pickle.loads(rows[1])
        
    def getAllData(self):    
        rows = asyncio.run( self.database.fetch_all(query="SELECT * FROM results ",
                       values={}) )
        return rows
    
    async def getAllDataNB(self):    
        return await self.database.fetch_all(query="SELECT * FROM results ",
                       values={}) 


         
    