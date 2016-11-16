
from jinja2 import Template,Environment,FileSystemLoader
import commands
import os
import tempfile

num_workers=6

z=os.environ["PROJECT_ZONE"].split("-")[0]
project = z+".gcr.io/"+os.environ["PROJECT_ID"]

def label_nodes(n,name):
    os.system("kubectl label nodes --overwrite %s app=%s" % (nodes[n], name))

# get GCP nodes info

c=commands.getoutput("kubectl get nodes")
nodes=[x.split(" ")[0] for x in c.split("\n")][1:]

# label nodes

print "Labeling nodes..."
label_nodes(0,"if")
label_nodes(1,"master")
for i in range(num_workers):
   label_nodes(i+2, ("worker%d"% i))

# Create Pods

print "Creating pods..."

def create_pods(name,imagename,port,com):
    obj={}
    obj["name"]  = name
    obj["image"] = project+"/"+imagename
    obj["com"]   = com.split(" ")
    obj["port"]  = port

    env = Environment(loader=FileSystemLoader('./template', encoding='utf-8'))
    tmpl = env.get_template("podspec_template.yml")

    f=tempfile.NamedTemporaryFile(mode="w",suffix=".yaml")

    f.write(tmpl.render(obj))
    f.flush()

    #print open(f.name).read()

    kcom="kubectl create -f "+f.name
    #print kcom
    os.system(kcom)
    f.close()


## if(interface) pod

create_pods("if","tf_if",8888,"/usr/local/bin/jupyter notebook")

# create TensorFlow cluster spec

cs="master|master:2222"
for i in range(num_workers):
   cs+=",worker%d|worker%d:2222" % (i,i)

def create_tf_pods(name):
    global cs
    com="/bin/grpc_tensorflow_server --cluster_spec=%s --job_name=%s --task_index=0" % (cs,name)
    create_pods(name,"tf_server", 2222 , com)

## master pod

create_tf_pods("master")

## worker pods

for i in range(num_workers):
    create_tf_pods("worker%d" % i)

