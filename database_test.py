import pymongo
def dbsetting():
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    db = myclient['BBC_NEWS']

    collection1 = db['Student_Account']
    collection2 = db['Subject_Info']
def insert_student(post,collection):
    result = collection.find_one({'ID': post['ID']})
    print(result)
    if result is not None:
        print("Username existed!")
    else:
        collection.insert_one(post)
        print("Successfully Insert")
def preprocess(file):
    with open(file) as infile, open("Dataset/outfile.csv", "w") as outfile:
        for line in infile:
            outfile.write(line.replace(",", " "))
    outfile_name = "Dataset/outfile.csv"
    return outfile_name

def csv_insert(file,colletion):

    data = open(file).read()
    for i, line in enumerate(data.split("\n")):
        if len(line.split()) > 0:
            content = line.split()

            post = {
                'Label': content[0],
                'Content': " ".join(content[1:])
            }
            print(post)
            colletion.insert_one(post)
    print("Insert Complete!")

def LabelDecoder(colletion):
     colletion.update_many({"Predict_Label": 0}, {"$set": {"Predict_Label": "business"}})
     colletion.update_many({"Predict_Label": 1}, {"$set": {"Predict_Label": "entertainment"}})
     colletion.update_many({"Predict_Label": 2}, {"$set": {"Predict_Label": "politics"}})
     colletion.update_many({"Predict_Label": 3}, {"$set": {"Predict_Label": "sport"}})
     colletion.update_many({"Predict_Label": 4}, {"$set": {"Predict_Label": "tech"}})
     colletion.update_many({"Actual_Label": 0}, {"$set": {"Actual_Label": "business"}})
     colletion.update_many({"Actual_Label": 1}, {"$set": {"Actual_Label": "entertainment"}})
     colletion.update_many({"Actual_Label": 2}, {"$set": {"Actual_Label": "politics"}})
     colletion.update_many({"Actual_Label": 3}, {"$set": {"Actual_Label": "sport"}})
     colletion.update_many({"Actual_Label": 4}, {"$set": {"Actual_Label": "tech"}})

def keyrename(db):
    db.Subject_Info.update_many({},{'$rename':{'Subject Group':'Subject_Group'}})
    db.Subject_Info.update_many({},{'$rename':{'Subject Title':'Subject_Title'}})
    db.Subject_Info.update_many({},{'$rename':{'Subject Group Code':'Subject_Group_Code'}})
    db.Subject_Info.update_many({},{'$rename':{'Component Code':'Component_Code'}})
    db.Subject_Info.update_many({},{'$rename':{'For_Every_(Week)':'For_Every_Week'}})
    db.Subject_Info.update_many({},{'$rename':{'Start Week':'Start_Week'}})
    db.Subject_Info.update_many({},{'$rename':{'End Week':'End_Week'}})
    db.Subject_Info.update_many({},{'$rename':{'Day of Week':'Day_of_Week'}})
    db.Subject_Info.update_many({},{'$rename':{'Start Time':'Start_Time'}})
    db.Subject_Info.update_many({},{'$rename':{'End Time':'End_Time'}})
    db.Subject_Info.update_many({},{'$rename':{'Venue':'Venue1'}})
    db.Subject_Info.update_many({},{'$rename':{'Teaching Staff':'Teaching_Staff'}})
    db.Subject_Info.update_many({},{'$rename':{'Remark':'Remark1'}})

def prerequisite():
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    db = myclient['SUBJECT_INFO']


def main():
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    db = myclient['SUBJECT_INFO']



    collection_set = {
        'Act' : db['Student_Account'],
        'Sbj_Info': db['Subject_Info'],
        'BBC': db['BBC_News'],
        'Paper': db['Paper'],
        'Student_Sbj': db['Student_Subject']
    }
    post ={
        'ID': '14111582D',
        'pwd': 'qwertyuiop[]'
    }

    Yzt_sbj = [
        {
            'Student_id': '14111515d',
            'Subject_Group': 'ENG3004',
            'Subject_Title': 'SOCIETY AND THE ENGINEERING',
            'Subject_Group_Code': '1005',
            'Component_Code': 'LTL001',
            'Day_of_Week': 'Tue',
            'Start_Time': '9:30',
            'End_Time': '12:20',
            'Venue': 'V312',
            'Teaching_Staff': 'TBD'
        },
        {
            'Student_id': '14111515d',
            'Subject_Group': 'ENG3004',
            'Subject_Title': 'SOCIETY AND THE ENGINEERING',
            'Subject_Group_Code': '1005',
            'Component_Code': 'SEM001',
            'Day_of_Week': 'Sat',
            'Start_Time': '13:30',
            'End_Time': '16:20',
            'Venue': 'TU201',
            'Teaching_Staff': 'TBD'
        },

    ]
    PaperDataset = [
        {
            'keywords': 'linear system',
            'Author': 'David G. Luenberger',
            'Title': 'Observing the State of a Linear System',
            'url':'https://ieeexplore.ieee.org/abstract/document/4323124'
        },
        {
            'keywords': 'linear system',
            'Author': 'Lili Wang,A. Stephen Morse',
            'Title': 'A Distributed Observer for a Time-Invariant Linear System',
            'url':'https://ieeexplore.ieee.org/abstract/document/8093658/authors#authors'
        },
        {
            'keywords': 'linear system',
            'Author': 'Chi-Tsong Chen',
            'Title': 'Linear system theory and design',
            'url':'https://julac.hosted.exlibrisgroup.com/primo-explore/fulldisplay?docid=HKPU_IZ21167270100003411&context=L&vid=HKPU&search_scope=BOOKS&isFrbr=true&tab=default_tab&lang=en_US'
        },
        {
            'keywords': 'website design',
            'Author': 'Sean D.Young,Renee Garett,Jason Chiu',
            'Title': 'A Literature Review: Website Design and User Engagement',
            'url':'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4974011/'
        },
        {
            'keywords': 'website design',
            'Author': 'Lulu Cai, Xiangzhen He, Yugang Dai, Kejian Zhu',
            'Title': 'A Literature Review: Website Design and User Engagement',
            'url':'https://iopscience.iop.org/article/10.1088/1742-6596/1087/6/062043/pdf'
        },
        {
            'keywords': 'website design',
            'Author': 'William C., McDowella, Rachel C. Wilsonb, Charles OwenKileJrc',
            'Title': 'An examination of retail website design and conversion rate',
            'url':'https://www.sciencedirect.com/science/article/abs/pii/S014829631630203X'
        }

    ]



    targetfile = "Dataset/bbc-text.csv"
    outfile = preprocess(targetfile)

    # collection_set['Paper'].delete_many({})
    # collection_set['Paper'].insert_many(PaperDataset)
    collection_set['Student_Sbj'].delete_many({})
if  __name__  ==   '__main__'  :
    main()
