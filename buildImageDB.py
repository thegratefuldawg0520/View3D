import sqlite3


def buildDBTable(dbPath,dbString):
	
	conn = sqlite3.connect(dbPath)
	curr = conn.cursor()

	curr.execute(dbString)
		
	conn.commit()
	conn.close()

dbPath = '/home/doopy/Documents/View3D/View3D_0_1/ImageMetadata.db'
dbString = """CREATE TABLE IF NOT EXISTS freidburg_xyz(

				img PRIMARY KEY,
				time,
				tx,
				ty,
				tz,
				qx,
				qy,
				qz,
				qw
);"""

buildDBTable(dbPath,dbString)
