import matplotlib.pyplot as plt

epoch_list = []
psnr_list = []

filepath ='./epoch2000_no/result_initial.txt'
f = open(filepath, 'r')
records = f.read().split('\n')
records = records[:-1]
for record in records:
    record = record.split(',')
    epoch_list.append(record[0])
    psnr_list.append(record[1])

f.close()

'''
# 1. train loss vs test loss for (1), (2)
plt.plot(epoch_list, train_list1, color='red', linewidth=2, label='OneSR')
plt.plot(epoch_list, train_list2, color='blue', linewidth=2, label='OneSR+BN')
plt.xlabel('epoch')
plt.ylabel('NLL loss')
plt.gca().invert_yaxis()
plt.legend()
plt.savefig('train_loss.png')
plt.close()
'''
plt.plot(epoch_list, psnr_list)
plt.xlabel('epoch')
plt.ylabel('PSNR')
plt.gca().invert_yaxis()
plt.savefig('test_loss.png')
plt.close()
