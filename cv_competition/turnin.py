# 1: Preprocessing

transform = transforms.Compose([
transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, target_transform=None, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

# 2: Model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=2)
        self.relu1=nn.ELU()
        nn.init.xavier_uniform(self.cnn1.weight)
        self.maxpool1=nn.MaxPool2d(kernel_size=2)
        self.cnn2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.relu2=nn.ELU()
        nn.init.xavier_uniform(self.cnn2.weight)
        self.maxpool2=nn.MaxPool2d(kernel_size=2)
#         self.cnn3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2)
#         self.relu3=nn.ELU()      
#         nn.init.xavier_uniform(self.cnn3.weight)
#         self.maxpool3=nn.MaxPool2d(kernel_size=2)
#         self.cnn4 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=5,stride=1,padding=2)
#         self.relu4=nn.ELU()
#         nn.init.xavier_uniform(self.cnn4.weight)
#         self.maxpool4=nn.MaxPool2d(kernel_size=2)
        self.fcl=nn.Linear(32*7*7,10)

    def forward(self, x):
        out=self.cnn1(x)
        out=self.relu1(out)
        out=self.maxpool1(out)
        out=self.cnn2(out)
        out=self.relu2(out)
        out=self.maxpool2(out)
#         out=self.cnn3(out)
#         out=self.relu3(out)
#         out=self.maxpool3(out)
#         out=self.cnn4(out)
#         out=self.relu4(out)
#         out=self.maxpool4(out)`
        out=out.view(out.size(0),-1)
        out=self.fcl(out)
        batch = list(x.size())[0]
        return out


net = Net()
# 3: Postprocess
epochs = 10
criterion=nn.CrossEntropyLoss()
learning_rate=0.015
optimizer=torch.optim.SGD(net.parameters(),lr=learning_rate)

number = len(train_loader)
for epoch in range(epochs): 

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        outputs = outputs.squeeze().float()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i == (number - 10):
            loss = running_loss / (number - 10)
            print("Epoch: {0}, Loss: {1}".format(epoch + 1, loss))
            running_loss = 0.0

print('Finished Training')
directory = './saved_model/'
if not os.path.exists(directory):
    os.makedirs(directory)
torch.save(net.state_dict(), './saved_model/model.pth')

directory = './saved_model/'
if not os.path.exists(directory):
    os.makedirs(directory)
torch.save(net.state_dict(), './saved_model/model.pth')
net = Net()
net.load_state_dict(torch.load('./saved_model/model.pth'))
net.eval()
f= open("result.txt","w+")
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, target_transform=None, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=2)

corrects = 0
total = 0
with torch.no_grad():
    for j, loader in enumerate([test_loader]):
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            inputs=Variable(inputs)
            outputs=net(inputs)
            _,predicted=torch.max(outputs.data,1)
            sm = torch.nn.Softmax() 
            probabilities = sm(outputs)
            thresh = 0.7
            for k in probabilities:
                k = torch.max(k)
                total += 1
                if k >= thresh:
                    k = 0
                    print(k)
                    f.write(str(k))
                    if i < sub_lim:
                        corrects += 1
                else:
                    k = 1
                    print(k)
                    f.write(str(k))
                    if i > sub_lim:
                        corrects += 1
            #Converted to probabilities
            # get the inputs
            # If needed, can add a line here that thresholds, or does something with the
            # 'outputs' variable to get it into a compatible format. An example is something
            # like _, outputs = torch.max(outputs.data, 1). If this line is needed, please
            # this along with your model architecture code labeled as 'postprocess'
#             corrects[j] += (predicted==labels).sum()
#             totals[j] += (labels.size(0))
            
f.close() 
accs = corrects / total
print("Accuracy for test in distribution data:", accs)

# 4: Written explanation

# Our code is still high based on the sample code rather than making many individual changes in strutucre and key logics. For the model, we try to apply CNN layer checking to process the train-data.
# However, we faced several problems because of the mismatching size probelms. @source :https://github.com/AbhirajHinge/CNN-with-Fashion-MNIST-dataset/blob/master/FashionMNISTwithCNN.py
# in the Fashion MNIST github, we found a person who also shares the similar key concepst about CNN layer chekcing. Rather than 4, we tried 2 CNN layer checking and got a accuracy around 85% - 90%
