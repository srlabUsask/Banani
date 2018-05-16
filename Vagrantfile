# -*- mode: ruby -*-
# vi: set ft=ruby :

#http://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-common/ClusterSetup.html

PRIVATE_KEY_SOURCE      = 'C:\Users\srlab\.vagrant.d\insecure_private_key'
PRIVATE_KEY_DESTINATION = '/home/vagrant/.ssh/id_rsa'
MASTER_IP               = '192.168.51.4'
Vagrant.configure("2") do |config|

  config.ssh.insert_key = false

  # define Master server
  config.vm.define "master" do |master|
    master.vm.hostname = "hadoop-master"
    master.vm.box = "ubuntu/trusty64"
    master.vm.synced_folder ".", "/home/vagrant/src", mount_options: ["dmode=775,fmode=664"]
    master.vm.network "private_network", ip: MASTER_IP
    master.vm.provider "virtualbox" do |v|
      v.name = "master"
      v.cpus = 1
      v.memory = 1024
    end
    # copy private key so hosts can ssh using key authentication (the script below sets permissions to 600)
    master.vm.provision :file do |file|
      file.source      = PRIVATE_KEY_SOURCE
      file.destination = PRIVATE_KEY_DESTINATION
    end
    master.vm.provision "shell", path: "bootstrap-master.sh"
  end

end
