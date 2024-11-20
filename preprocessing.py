# ########################### For Amazon-book and Gowalla Dataset ########################################
# with open("./Data/amazon-book/train1.txt","r") as f1, open("./Data/amazon-book/test1.txt","r") as f2, open("./Data/amazon-book/test2.txt","w") as f3:
#     items=set()
#     for l in f1:
#         if len(l)>0:
#             l=l.strip('\n').split(' ')
#             uid=l[0]
#             for i in l[1:]:
#                 items.add(i)
#     for l in f2:
#         if len(l)>0:
#             l=l.strip('\n').split(' ')
#             uid=l[0]
#             items_test=[]
#             for i in l[1:]:
#                 if i in items:
#                     items_test.append(i)
#             f3.write(uid + " " + " ".join(items_test) + "\n")
                



# with open("./Data/amazon-book/train1.txt","r") as f1, open("./Data/amazon-book/test3.txt","w") as f2,open("./Data/amazon-book/test2.txt","r") as f3, open("./Data/amazon-book/train3.txt","w") as f4,open("./Data/amazon-book/train1.txt","r") as f5:
#     items=set()
#     for l in f1:
#         if len(l)>0:
#             l=l.strip('\n').split(' ')
#             uid=l[0]
#             for i in l[1:]:
#                 items.add(i)
#     items_index=dict(zip(items,range(1,len(items)+1)))
#     # print(items_index)
#     for l in f5:
#         if len(l)>0:
#             l=l.strip('\n').split(' ')
#             uid=l[0]
#             items_train=[str(items_index[i]) for i in l[1:] if i in items]
#             f4.write(uid + " " + " ".join(items_train) + "\n")

#     for l in f3:
#         if len(l)>0:
#             l=l.strip('\n').split(' ')
#             uid=l[0]
#             items_test=[str(items_index[i]) for i in l[1:] if i in items]
#             f2.write(uid + " " + " ".join(items_test) + "\n")


#################### For ml-100k dataset ############################
# import pandas as pd
# import csv

# # Load the .base file, assuming space-separated values
# df = pd.read_csv('./Data/ml-100k/u1.test', delim_whitespace=True, header=None)

# # Manually assign column names
# df.columns = ['userid', 'itemid', 'rating', 'timestamp']

# # Extract only the 'userid' and 'itemid' columns
# df = df[['userid', 'itemid']]

# # Group by 'userid' and aggregate 'itemid' as a list concatenated into a string
# grouped_df = df.groupby('userid')['itemid'].apply(lambda x: ' '.join(x.astype(str).values)).reset_index()

# # Rename the columns if needed
# grouped_df.columns = ['userid', 'items']

# # Save the result to a new .txt file with space-separated values
# grouped_df.to_csv('./Data/ml-100k/test.txt', sep=' ', index=False, header=False)


### For ml 1m dataset ###
# import random

# # Read dataset
# with open('./Data/ml-1m/ratings.dat', 'r') as file:
#     data = file.readlines()

# user_items = {}

# # Parse the dataset (userid::itemid::rating::time format)
# for line in data:
#     userid, itemid, rating, timestamp = line.strip().split('::')
    
#     if userid not in user_items:
#         user_items[userid] = []
    
#     user_items[userid].append(itemid)

# # Split into train and test sets
# train_data = []
# test_data = []

# for userid, items in user_items.items():
#     # Shuffle the items to randomize the split
#     random.shuffle(items)
    
#     # Calculate split point for 80-20
#     split_index = int(0.8 * len(items))
    
#     # Train set - 80%
#     train_items = items[:split_index]
#     train_data.append(f"{userid} {' '.join(train_items)}\n")
    
#     # Test set - 20%
#     test_items = items[split_index:]
#     if test_items:  # Only include users who have items left for testing
#         test_data.append(f"{userid} {' '.join(test_items)}\n")

# # Write to train and test files
# with open('./Data/ml-1m/train.txt', 'w') as train_file:
#     train_file.writelines(train_data)

# with open('./Data/ml-1m/test.txt', 'w') as test_file:
#     test_file.writelines(test_data)

# print("Train and test files created successfully!")
