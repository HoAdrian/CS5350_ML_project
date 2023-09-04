from adaboost import *



np.random.seed(2021)

train_data = np.array([[1.0, 1.0],[-1.0,0.0]])
labels = np.array([-1.0,-1.0])
num_data = len(train_data)
print(train_data.shape)
w = np.array([2,3])
y = np.matmul(train_data,w)
mask_positive = (y>=0).astype(int)
mask_negative = (y<0).astype(int)
mask_negative = mask_negative*-1
out = mask_positive + mask_negative
print("out ", out)
print("compare ", out==labels)
print("accuracy ", sum(out==labels)/num_data)

# train_data = np.array([[1.0, 1.0, 1.0],[-1.0, 0.0, -1.0]])
# test_data = np.array([[3.0, 1.0, 1.0],[-2.0, -7.0, -1.0]])
# ensemble = Adaboost(train_data, num_learner=1)
# ensemble.get_weak_learners(train_data, test_data)
# ensemble.adaboost(num_iter=100, dev_data=test_data)