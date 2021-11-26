import sys
sys.path.append("..")

from helpers import *
from models import *


def fill_absolute_floor(row_image_list) -> bool:
    """
    Usage:
        RowImage list 넣으면, absolute floor 찾아서 무료로 채워드립니다.
    """
    sources = [] # all sources in row_image_list
    for row_image in row_image_list:
        source = row_image.source
        if not source in sources:
            sources.append(source)

    relation_table = [[None for _ in range(len(sources))] for _ in range(len(sources))] # 방향그래프 인접 테이블 // source{i}_floor0의 위치 = source{j}_floor0의 위치 + Edge(i,j)
    
    for i in range(len(row_image_list)):
        for j in range(i):
            row_i = row_image_list[i]
            row_j = row_image_list[j]

            if match_row(row_i, row_j):
                source_i_index = sources.index(row_i.source)
                source_j_index = sources.index(row_j.source)
                relative_floor_i = row_i.relative_floor
                relative_floor_j = row_j.relative_floor
                relation_table[source_i_index][source_j_index] = relative_floor_j - relative_floor_i
                relation_table[source_j_index][source_i_index] = relative_floor_i - relative_floor_j

    source_floor_map = dict() # Source -> int
    match_success = True
    flag_list = [False for _ in range(len(sources))]
    
    queue = [0]
    flag_list[0] = True
    source_floor_map[sources[0]] = 0
    
    while len(queue) > 0:
        curr_index = queue.pop()

        for i in range(len(sources)):
            if relation_table[i][curr_index] == None : continue
            if flag_list[i] : continue
            
            source_floor_map[sources[i]] = source_floor_map[sources[curr_index]] + relation_table[i][curr_index]
            flag_list[i] = True
            queue.append(i)

    assert all([flag for flag in flag_list]), "매칭 실패!! ㅠㅠ"

    for row_image in row_image_list:
        row_image.absolute_floor = source_floor_map[row_image.source] + row_image.relative_floor
        

if __name__ == "__main__":
    source_list = []
    row_image_list = []
    
    for i in range(7):
        path = "./data/source/source{}.png".format(i)
        img = cv2.imread(path)
        source = Source(img, path)
        source_list.append(img)

        for j in range(2):
            img = cv2.imread("./data/row_image/source{}_floor{}.png".format(i, j))
            row_image = RowImage(img, source, j)
            row_image_list.append(row_image)

    print("========== before ==========")
    for row_image in row_image_list:
        print(row_image)
        
    fill_absolute_floor(row_image_list)

    
    print("========== after ==========")
    for row_image in row_image_list:
        print(row_image)


                               
