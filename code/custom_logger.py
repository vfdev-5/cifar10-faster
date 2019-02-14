

class TSVLogger(object):

    def __init__(self):
        self.log = ['epoch\thours\ttop1Accuracy']

    def append(self, output):
        epoch, hours, acc = output['epoch'], output['total time'] / 3600, output['test acc'] * 100
        self.log.append('{}\t{:.8f}\t{:.2f}'.format(epoch, hours, acc))

    def __str__(self):
        return '\n'.join(self.log)


class TableLogger(object):

    def append(self, output):
        if not hasattr(self, 'keys'):
            self.keys = output.keys()
            print(*('{k:>12s}'.format(k=k) for k in self.keys))
        filtered = [output[k] for k in self.keys]
        print(*('{v:12.4f}'.format(v=v) if isinstance(v, float) else '{v:12}'.format(v=v) for v in filtered))
