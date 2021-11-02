import config_audio
import os

src_pth = "/paras/paras/Music/DNN_model_100_species/data"
dest_pth = "/paras/paras/Music/DNN_model_100_species/data_sr"

for clas in config_audio.class_names:
    src_dir = os.path.join(src_pth,clas)
    dest_dir = os.path.join(dest_pth,clas)
    try:
        os.makedirs(dest_dir,0o775)
    except FileExistsError:
        print("ALredy Exists")
    
    for root, dir, file in os.walk(src_dir):
        for f in file:
            old_file = os.path.join(src_dir,f)
            new_file = os.path.join(dest_dir,f)
            print(old_file +"::"+ new_file)
            os.system("sox "+old_file+" -r 41000 -c 1 "+ new_file)