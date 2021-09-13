model_dict = {
    'MobileNetv2' : 1280,
    #'MobileNetv3_small' : 576,
    #'MobileNetv3_large' : 960,
    'Inception' : 2048,
    'ShuffleNet' : 1024,
    'ResNet34' : 512,
    'ResNet50' : 2048
}

def get_architecture(args):
    output_channel =0
    in_channel_1 = 1024
    in_channel_2 = 512

    if args.arch in ['MobileNetv2']:
        net = mobilenet_v2(pretrained = True).to(args.device)
        output_channel = 1280
        del net.classifier
        layer_count = 0
        for child in net.children():
            layer_count+=1
            if layer_count< int(len(list(net.children()))* args.freeze):
                for param in child.parameters():
                    param.requires_grad=False
        net.classifier = nn.Sequential(
                        nn.Linear(output_channel, in_channel_1),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(in_channel_1, in_channel_2),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(in_channel_2,args.num_classes),
                        )
    
    elif args.arch in ['MobileNetv3']:
        net = mobilenet_v3_large(pretrained = True).to(args.device)
        output_channel = 960

        del net.classifier
        layer_count = 0
        for child in net.children():
            layer_count+=1
            if layer_count< int(len(list(net.children()))* args.freeze):
                for param in child.parameters():
                    param.requires_grad=False
        net.classifier = nn.Sequential(
                        nn.Linear(output_channel, in_channel_1),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(in_channel_1, in_channel_2),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(in_channel_2,args.num_classes),
                        )
    
    elif args.arch in ['Inception']:
        net = inception_v3(pretrained = True).to(args.device)
        output_channel = 2048
        del net.fc
        layer_count = 0
        for child in net.children():
            layer_count+=1
            if layer_count< int(len(list(net.children()))* args.freeze):
                for param in child.parameters():
                    param.requires_grad=False
        net.fc = nn.Sequential(
                        nn.Linear(output_channel, in_channel_1),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(in_channel_1, in_channel_2),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(in_channel_2,args.num_classes),
                        )

    elif args.arch in ['ShuffleNet']:
        net = shufflenet_v2_x1_0(pretrained = True).to(args.device)
        output_channel = 1024

        del net.fc
        layer_count = 0
        for child in net.children():
            layer_count+=1
            if layer_count< int(len(list(net.children()))* args.freeze):
                for param in child.parameters():
                    param.requires_grad=False
        net.fc = nn.Sequential(
                        nn.Linear(output_channel, in_channel_1),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(in_channel_1, in_channel_2),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(in_channel_2,args.num_classes),
                        )
    
    elif args.arch in ['ResNet34']:
        net = resnet34(pretrained=True).to(args.device)
        output_channel = 512

        del net.fc
        layer_count = 0
        for child in net.children():
            layer_count+=1
            if layer_count< int(len(list(net.children()))* args.freeze):
                for param in child.parameters():
                    param.requires_grad=False
        net.fc = nn.Sequential(
                        nn.Linear(output_channel, in_channel_1),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(in_channel_1, in_channel_2),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(in_channel_2,args.num_classes),
                        )

    elif args.arch in ['ResNet50']:
        net = resnet50(pretrained=True).to(args.device)
        output_channel = 2048
        del net.fc
        layer_count = 0
        for child in net.children():
            layer_count+=1
            if layer_count< int(len(list(net.children()))* args.freeze):
                for param in child.parameters():
                    param.requires_grad=False
        net.fc = nn.Sequential(
                        nn.Linear(output_channel, in_channel_1),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(in_channel_1, in_channel_2),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(in_channel_2,args.num_classes),
                        )

    return net