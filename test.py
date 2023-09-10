with open("data/new_pd_wip/v6/pd_wip_3d_test_new.txt", 'r') as f:
    data = f.readlines()
    
test_case_name = [line.strip() for line in data]

with open("data/new_pd_wip/v6/pd_wip_3d.txt", 'r') as f:
    data = f.readlines()
all_data = [line.strip() for line in data]


train_data = []
test_data = []
for line in all_data:
    is_test = False
    for t_case_name in test_case_name:
        if t_case_name in line:
            is_test = True
    if is_test:
        test_data.append(line)
    else:
        train_data.append(line)
train_data = [line + "\n" for line in train_data] 
test_data = [line + "\n" for line in test_data] 

with open("data/new_pd_wip/v6/pd_wip_3d_test.txt", 'w') as f:
    f.writelines(test_data)
    
with open("data/new_pd_wip/v6/pd_wip_3d_train.txt", 'w') as f:
    f.writelines(train_data)