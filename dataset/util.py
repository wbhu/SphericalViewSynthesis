import os
import os.path


def translate_datalist(data_root=None, data_list=None, number=0):
    assert number > 0
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples.".format(len(list_read)))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if number == 1:
            if len(line_split) != number:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = image_name  # just set place holder for label_name, not for use
        elif number == 2:
            if len(line_split) != number:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = os.path.join(data_root, line_split[1])
        else:
            raise (RuntimeError("No support number error : " + str(number) + "\n"))
        item = (image_name, label_name)
        image_label_list.append(item)
    return image_label_list
