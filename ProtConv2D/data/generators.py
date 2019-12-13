from keras.utils import Sequence


class H5_Image_Generator(Sequence):
    def __init__(
        self,
        indices,
        inputs,
        outputs,
        batch_size,
        is_train,
        shuffle=True,
        crops_per_image=0,
        crop_width=32,
        crop_height=32,
    ):
        self.nsamples = inputs[inputs.keys()[0]].shape[0]
        self.is_train = is_train

        print("n samples:", self.nsamples)
        self.indices = indices
        print("H5_Image_Generator")
        print([inputs[i].shape for i in inputs], [outputs[o].shape for o in outputs])

        self.inputs = {
            i: np.take(a=inputs[i], indices=self.indices, axis=0) for i in inputs
        }
        self.outputs = {
            o: np.take(a=outputs[o], indices=self.indices, axis=0) for o in outputs
        }

        print(
            [self.inputs[i].shape for i in self.inputs],
            [self.outputs[o].shape for o in self.outputs],
        )
        self.batch_size = batch_size
        self.crops_per_image = crops_per_image
        self.effective_batch_size = (
            self.batch_size
            if crops_per_image <= 0
            else self.batch_size / self.crops_per_image
        )

        print("batch size ", self.batch_size)
        print("n crops ", self.crops_per_image)
        print("eff bs ", self.effective_batch_size)
        self.shuffle = shuffle

        self.crop_width = crop_width
        self.crop_height = crop_height
        self.on_epoch_end()

    def __len__(self):
        return np.ceil(len(self.indices) / float(self.effective_batch_size))

    # def on_epoch_end(self):
    #    'Updates indexes after each epoch'
    #    self.indexes = np.arange(len(self.list_IDs))
    #    if self.shuffle == True:
    #        np.random.shuffle(self.indexes)

    def random_crop(self, img, random_crop_size):
        # Note: image_data_format is 'channel_last'
        assert img.shape[2] == 3
        height, width = img.shape[0], img.shape[1]
        dy, dx = random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return img[y : (y + dy), x : (x + dx), :]

    # def __data_generation(self, list_IDs_temp):
    #     'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    #     # Initialization
    #     X = np.empty((self.batch_size, *self.dim, self.n_channels))
    #     y = np.empty((self.batch_size), dtype=int)

    #     # Generate data
    #     for i, ID in enumerate(list_IDs_temp):
    #         # Store sample
    #         X[i,] = np.load('data/' + ID + '.npy')

    #         # Store class
    #         y[i] = self.labels[ID]

    #     return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __getitem__(self, idx):
        batches_x = {}
        for inp in self.inputs:

            batches_x[inp] = self.inputs[inp][
                idx * self.effective_batch_size : (idx + 1) * self.effective_batch_size,
            ]
            if self.crops_per_image > 0:
                # print("batch",inp, batches_x[inp].shape)
                if inp == "main_input":
                    batch_crops = np.ndarray(
                        (self.batch_size, self.crop_width, self.crop_height, 3)
                    )
                    # print (batch_crops.shape)

                    for i1 in range(self.effective_batch_size):
                        for i2 in range(self.crops_per_image):
                            img = batches_x[inp][i1, :, :, :]
                            # print("img ", img.shape)
                            crop = self.random_crop(
                                img, (self.crop_width, self.crop_height)
                            )
                            # print("crop ",i1,i2,crop.shape)
                            batch_crops[i1 + i2, :, :] = crop
                    # print("old batch:",batches_x[inp].shape)
                    # print("new batch:",batch_crops.shape)
                    batches_x[inp] = batch_crops
                else:
                    shape = batches_x[inp].shape
                    shape = tuple([self.batch_size] + list(shape[1:]))

                    batch_tmp = np.ndarray(shape)
                    # print(inp, shape, batch_tmp.shape)
                    for i1 in range(self.effective_batch_size):
                        for i2 in range(self.crops_per_image):
                            batch_tmp[i1 + i2,] = batches_x[inp][i1]
                    # print("old batch:",batches_x[inp].shape)
                    # print("new batch:",batch_tmp.shape)
                    batches_x[inp] = batch_tmp
            # print("input" ,inp, idx, batches_x[inp].shape, self.is_train)
            # assert len(batches_x[i]) > 0

        batches_y = {}
        for o in self.outputs:
            batches_y[o] = self.outputs[o][
                idx * self.effective_batch_size : (idx + 1) * self.effective_batch_size,
            ]
            if self.crops_per_image > 0:
                shape = batches_y[o].shape
                shape = tuple([self.batch_size] + list(shape[1:]))

                batch_tmp = np.ndarray(shape)
                # print(o, shape, batch_tmp.shape)
                for i1 in range(self.effective_batch_size):
                    for i2 in range(self.crops_per_image):
                        batch_tmp[i1 + i2,] = batches_y[o][i1]
                # print("old batch:",batches_y[o].shape)
                # print("new batch:",batch_tmp.shape)
                batches_y[o] = batch_tmp
            # print("output",o, idx, batches_y[o].shape, self.is_train)
            # assert len(batches_y[o]) > 0

        return (batches_x, batches_y)
