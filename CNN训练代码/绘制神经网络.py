import sys
def makeDefConfig():
    args = dict()
    #path
    path = sys.argv[0]
    print(path)
    # if isWindowsSystem():
    # last = path.rindex('\\')
    # else:
    last = path.rindex('/')

    path = path[0:last+1]
    args.update({'input':path + 'nn.gv'})
    args.update({'output':path + 'nn.png'})
    #visual nodes number
    args.update({'visual_num':10})
    #layers config:(name,nodes number,color)
    args.update({'layers_cfg':(('input',10,'blue4'),('h1',7,'red2'),('h2',5,'red2'),('h3',4,'red2'),('out',2,'seagreen2'))})
    layers = args['layers_cfg']
    args.update({'layers_num':len(layers)})
    #connects:layer_i->lay_j
    args.update({'connects':([0,1],[1,2],[2,3],[3,4])})
    return args

print(makeDefConfig())