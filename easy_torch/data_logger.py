
import os
import json

class DataLogger(object):
    def __init__(self, folder_name, log_freq=10, test_freq=100):
        """
        folder_name: (str) The name of the log folder
        """
        self.folder_name = folder_name
        if not folder_name is None:
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)

            if not os.path.exists(os.path.join(folder_name, "res.json")):
                self.res = {}
        
            else:
                with open(os.path.join(folder_name, "res.json"), "r") as f:
                    self.res = json.load(f)

        self.log_freq = log_freq
        self.test_freq = test_freq
        self.train_monitor = Monitor()
        self.test_monitor = Monitor()
        self.test_acc = 0
        self.test_loss = 0
        self.train_acc = 0
        self.train_loss = 0
        self.previous_iteration = 0

    def update(self, name, iteration, *args, force_log=False):
        getattr(self, "%s_monitor" % name).update_metrics(*args)
        if (name != "test" and self.require_log(iteration)) or force_log:
            acc, loss = getattr(self, "%s_monitor" % name).return_metrics()
            setattr(self,"%s_acc" % name, acc) 
            setattr(self,"%s_loss" % name, loss) 
            self._logg(acc, "%s_acc" % name, iteration)
            self._logg(loss, "%s_loss" % name, iteration)
            setattr(self, "%s_monitor" % name, Monitor())

    def _logg(self, value, name, iteration):
        if hasattr(self, "res"):
            if str(iteration) in self.res:
                self.res[str(iteration)][name] = value

            else:
                self.res[str(iteration)] = {name:  value}

    def require_log(self, iteration, name="log"):
        ans = int(iteration/getattr(self, "%s_freq" % name)) !=\
                int(self.previous_iteration/getattr(self, "%s_freq" % name))

        return ans

    def save(self):
        if hasattr(self, "folder_name"):
            with open(os.path.join(self.folder_name, "res.json"), "w") as f:
                json.dump(self.res, f, indent=4)

class Monitor(object):
    def __init__(self):
        self.acc = 0
        self.loss = 0 
        self.count = 0

    def update_metrics(self, loss, output, target):
        self.loss += loss.item()
        self.count += 1

    def return_metrics(self):
        return self.acc/self.count, self.loss/self.count

        

