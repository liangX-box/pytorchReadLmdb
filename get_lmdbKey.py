import sys, lmdb
caffe_root = "/usr/caffe"
sys.path.insert(0, caffe_root+'python')
import caffe

if __name__ == "__main__":
    lmdb_path = "/usr/train_lmdb/"
    saveKey_path = "/usr/train_key.txt"
    count = 0
    fl = open(saveKey_path, "w")
    lmdb_env = lmdb.open(lmdb_path)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    for key, value in lmdb_cursor:
        fl.write("%s\n" %(key))
        count += 1
    fl.close()
    print ("the number of samples is: %d" %(count))
    print ("process end!")

