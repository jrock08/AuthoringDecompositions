import os
import data.IntrinsicData_async as IntrinsicData_async
import data.ImageData_async as ImageData_async

def get_data_dir():

    train_root_dir = '/data/jrock/train_data'
    test_root_dir = '/data/jrock/test_data'

    shading_dir = train_root_dir + '/RenderedShading/'
    material_dir = train_root_dir + '/MaterialDetail/normalized/'
    mondrian_dir = train_root_dir + '/MondrianAlbedo/MondrianAlbedoOrig'

    vase_dir = test_root_dir + '/Vases'
    iiw_dir = test_root_dir + '/iiw'
    MIT_dir = test_root_dir + '/MIT_orig'
    MIT_augmented_dir = test_root_dir + '/MIT_augmented'

    return shading_dir, material_dir, mondrian_dir, vase_dir, iiw_dir, MIT_dir, MIT_augmented_dir

class Constants:
    def __init__(self, root_id, run_id):
        self.root_id = root_id
        self.run_id = run_id
        self.assign_data_dir()
        self.patch_size = 32
        self.batch_size = 50
        self.num_train_epochs = 100

    def assign_data_dir(self):
        self.shading_dir, self.material_dir, self.mondrian_dir, self.vase_dir, self.iiw_dir, self.MIT_dir, self.MIT_augmented_dir = get_data_dir()

    def root_dir(self):
        return '/data/jrock/out%s/'%(self.root_id)

    def summary_dir(self):
        return self.root_dir() + '/summary/%s'%(self.run_id)

    def model_dir(self):
        return self.root_dir() + '/model'

    def vae_out_dir(self):
        return self.root_dir() + '/vae/%s'%(self.run_id)

    def encode_dir(self):
        return self.root_dir() + '/encode'

    def pred_dir(self):
        return self.root_dir() + '/prediction/%s'%(self.run_id)

    def make_pred_dirs(self):
        all_dirs = [self.pred_dir(), self.encode_dir()]

        for d in all_dirs:
            if not os.path.isdir(d):
                os.makedirs(d)

    def make_dirs(self):
        print "making dirs at %s"%(self.root_dir())
        all_dirs = [self.root_dir(), self.model_dir(), self.vae_out_dir()]

        for d in all_dirs:
            if not os.path.isdir(d):
                os.makedirs(d)


    def default_image_data_shape(self):
        return [self.batch_size, self.patch_size, self.patch_size, 3]

    def get_augmented_data(self, patch_size = None, batch_size = None, num_epochs=None, dtype='train'):
        root_dir = self.MIT_augmented_dir
        patch_size = patch_size or self.patch_size
        batch_size = batch_size or self.batch_size
        num_epochs = num_epochs or -1

        shading_dir = root_dir + '/' + dtype + '/shading/'
        albedo_dir = root_dir + '/' + dtype + '/albedo/'
        image_dir = root_dir + '/' + dtype + '/images/'
        mask_dir = root_dir + '/' + dtype + '/masks/'

        return IntrinsicData_async.default_data(image_dir, shading_dir, albedo_dir,
                mask_dir, [patch_size]*2, batch_size,
                10*batch_size, 1210, same_process=True, num_epochs=num_epochs)

    def get_orig_datamgr(self, patch_size = None, batch_size = None, num_epochs=None, dtype='train'):
        root_dir = self.MIT_dir
        patch_size = patch_size or self.patch_size
        batch_size = batch_size or self.batch_size
        num_epochs = num_epochs or -1

        shading_dir = root_dir + '/' + dtype + '/shading/'
        albedo_dir = root_dir + '/' + dtype + '/reflectance/'
        image_dir = root_dir + '/' + dtype + '/image/'
        mask_dir = root_dir + '/' + dtype + '/mask/'

        return IntrinsicData_async.default_data(image_dir, shading_dir, albedo_dir,
                mask_dir, [patch_size]*2, 1,
                1, 10, same_process=True, num_epochs=num_epochs)[0]


    def get_vase_datamgr(self, patch_size = None, batch_size = None, num_epochs=None):
        root_dir = self.vase_dir
        patch_size = patch_size or self.patch_size
        batch_size = batch_size or self.batch_size
        num_epochs = num_epochs or -1

        image_dir = root_dir + '/image_out/'
        mask_dir = root_dir + '/mask_out/'

        return ImageData_async.default_data(image_dir,
                mask_dir, [patch_size]*2, 1,
                1, 8, same_process=True, num_epochs=num_epochs)[0]

    def get_iiw_datamgr(self, patch_size = None, batch_size = None, num_epochs=None):
        root_dir = self.iiw_dir
        patch_size = patch_size or self.patch_size
        batch_size = batch_size or self.batch_size
        num_epochs = num_epochs or -1

        image_dir = root_dir + '/images/'
        mask_dir = None

        return ImageData_async.default_data(image_dir,
                mask_dir, [patch_size]*2, 1,
                1, 2, same_process=True, num_epochs=num_epochs)[0]

    def get_rendered_shading_data(self, patch_size = None, batch_size = None, num_epochs=None):
        root_dir = self.shading_dir
        patch_size = patch_size or self.patch_size
        batch_size = batch_size or self.batch_size
        num_epochs = num_epochs or self.num_train_epochs

        shading_dir = root_dir + '/shading/'
        mask_dir = root_dir + '/mask/'

        return ImageData_async.default_data(shading_dir, mask_dir, [patch_size]*2, batch_size,
            10*batch_size, 70, same_process=True, num_epochs=num_epochs)[1]

    def get_mondrian_albedo_data(self, patch_size = None, batch_size = None, num_epochs=None):
        root_dir = self.mondrian_dir
        patch_size = patch_size or self.patch_size
        batch_size = batch_size or self.batch_size
        num_epochs = num_epochs or self.num_train_epochs

        shading_dir = root_dir + '/albedo/'
        mask_dir = root_dir + '/mask/'

        return ImageData_async.default_data(shading_dir, mask_dir, [patch_size]*2, batch_size,
            10*batch_size, 500, same_process=True, num_epochs=num_epochs)[1]

    def get_material_data(self, patch_size = None, batch_size = None, num_epochs=None):
        root_dir = self.material_dir
        patch_size = patch_size or self.patch_size
        batch_size = batch_size or 45
        num_epochs = num_epochs or self.num_train_epochs

        shading_dir = root_dir + '/material_detail/'
        mask_dir = root_dir + '/mask/'

        return ImageData_async.default_data(shading_dir, mask_dir, [patch_size]*2, batch_size,
            10*batch_size, 45, same_process=True, num_epochs=num_epochs)[1]

