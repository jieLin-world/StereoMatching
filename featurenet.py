import timm

all_pretrained_models_available = timm.list_models(pretrained=True)
f = open('feature_net.txt', 'a')
for i in range(591):
    f.write('%s\n' %(all_pretrained_models_available[i]))
print(all_pretrained_models_available)
print(len(all_pretrained_models_available))