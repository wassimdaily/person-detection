import sqlite3
import os
import sys


class Image(object):

    def __init__(self, dbname="Images.db"):
        self.image_name = []
        self.dbname = dbname

    def load_directory(self, path='/home/hydro/Pictures'):
        """
        :param path: Provide Path of File Directory
        :return: List of image Names
        """
        for x in os.listdir(path):
            self.image_name.append(x)

        return self.image_name

    def create_database(self, name, image):
        """
        :param name: String
        :param image:  BLOP Data
        :return: None
        """

        conn = sqlite3.connect(self.dbname)

        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS my_table 
        (name TEXT, Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, image BLOP)""")

        cursor.execute(""" insert into my_table(`name`,`image`) values(?,?)""",(name,image))

        cursor.close()

        conn.commit()
        conn.close()

def main():
    obj = Image()
    os.chdir("/home/hydro/Pictures")
    for x in obj.load_directory():

        if ".png" in x:
            with open(x,"rb") as f:
                data = f.read()
                obj.create_database(name=x, image=data)
                print("{} Added to database ".format(x))

        elif ".jpg" in x:
            with open(x,"rb") as f:
                data = f.read()
                obj.create_database(name=x,image=data)
                print("{} added to Database".format(x))


if __name__ == "__main__":
    main()

