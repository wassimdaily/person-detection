import sqlite3
import os
import sys
import time
from datetime import datetime
from datetime import date

now = datetime.now()
#start_time_time = datetime.now()
current_time = now.strftime("%H:%M:%S")

class Image(object):

    def __init__(self, dbname="Image.db"):
        self.image_name = []
        self.dbname = dbname

    def convertImagetobase64(self, image):
        if image != None:
            return b64encode(image).decode("utf-8")
        return image

    def load_directory(self, path="/home/hydro/person_detection-master/Pictures"):
        """
        :param path: Provide Path of File Directory
        :return: List of image Names
        """
        for x in os.listdir(path):
            self.image_name.append(x)

        return self.image_name

    def create_database(self, name, start_time, end_time, image, person):
        """
        :param name: String
        :param start_time: String
        :param end_time: String
        :param image:  BLOP Data
        :param person:  String
        :return: None
        """

        conn = sqlite3.connect(self.dbname)

        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS my_table 
        (name TEXT, start_time Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, end_time Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, image BLOP, person Object)""")

        cursor.execute(""" insert into my_table(`name`, `start_time`, `end_time`, `image`,`person`) values(?,?,?,?,?)""",(name, start_time, end_time, image, person))

        cursor.close()

        conn.commit()
        conn.close()

def main():
    obj = Image()
    os.chdir("/home/hydro/person_detection-master/Pictures")
    for x in obj.load_directory():

        if ".png" in x:
            with open(x,"rb") as f:
                data = f.read()
                obj.create_database(name=x, start_time=now, end_time=current_time, image=data, person='Person')
                print("{} Added to database ".format(x))

        elif ".jpg" in x:
            with open(x,"rb") as f:
                data = f.read()
                obj.create_database(name=x, start_time=now, end_time=current_time, image=data, person='Person')
                print("{} added to Database".format(x))


if __name__ == "__main__":
    main()



