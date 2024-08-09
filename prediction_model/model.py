import tensorflow as tf
import numpy as np
import os

class Model(tf.keras.Model):
    """Model class"""
    def __init__(self, inputs, targets, learning_rate, reg_constant, dropout_rate,
                 num_neurons, lr_initial, lr_decay_step, batch_size, model_name):

        super(Model, self).__init__()
        self.inputs = inputs
        self.targets = targets
        self.learning_rate = learning_rate
        self.reg_constant = reg_constant
        self.dropout_rate = dropout_rate
        self.num_layers = len(num_neurons)
        self.num_neurons = num_neurons
        self.lr_initial = lr_initial
        self.lr_decay_step = lr_decay_step
        self.batch_size = batch_size
        self.model_name = model_name

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # Building the model layers
        self.fc_layers = [tf.keras.layers.Dense(units=n, activation='relu',
                                                kernel_regularizer=tf.keras.regularizers.l2(reg_constant))
                          for n in num_neurons]
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.output_layer = tf.keras.layers.Dense(units=1, activation='relu', use_bias=False)

        # Define the loss function
        self.loss_fn = tf.keras.losses.MeanSquaredError()


    def get_config(self):
        config = super(Model, self).get_config()
        config.update({
            'inputs': self.inputs,
            'targets': self.targets,
            'learning_rate': self.learning_rate,
            'reg_constant': self.reg_constant,
            'dropout_rate': self.dropout_rate,
            'num_neurons': self.num_neurons,
            'lr_initial': self.lr_initial,
            'lr_decay_step': self.lr_decay_step,
            'batch_size': self.batch_size,
            'model_name': self.model_name
        })
        return config

    
    @classmethod
    def from_config(cls, config):
        return cls(
            inputs=config['inputs'],
            targets=config['targets'],
            learning_rate=config['learning_rate'],
            reg_constant=config['reg_constant'],
            dropout_rate=config['dropout_rate'],
            num_neurons=config['num_neurons'],
            lr_initial=config['lr_initial'],
            lr_decay_step=config['lr_decay_step'],
            batch_size=config['batch_size'],
            model_name=config['model_name']
        )


    def call(self, inputs, training=False):
        x = inputs
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        if training:
            x = self.dropout(x, training=training)
        return tf.reshape(self.output_layer(x), [-1])

    def compute_loss(self):
        log_targets = tf.math.log(1 + self.targets)
        log_predictions = tf.math.log(1 + self.call(self.inputs, training=True))
        mse_loss = tf.reduce_mean(self.loss_fn(log_targets, log_predictions))
        reg_loss = tf.reduce_sum(self.losses)  # This captures the regularization losses
        return mse_loss + reg_loss

    def train_step(self):
        with tf.GradientTape() as tape:
            loss = self.compute_loss()
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=0.1)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.global_step.assign_add(1)
        return loss

    def train(self, traindata, trainlabel, testdata, testlabel, num_train_steps):
        """Train the model for a number of steps, save checkpoints, write graph and loss for tensorboard"""
        print(os.path.abspath('./'))
        checkpoint_dir = f'./checkpoints/{self.model_name}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

        num_datapoints = traindata.shape[0]
        list_datapoints = np.arange(num_datapoints)
        num_batches = int(np.ceil(num_datapoints / self.batch_size))

        # Set up TensorBoard
        train_summary_writer = tf.summary.create_file_writer(f'./logs/train/{self.model_name}')
        test_summary_writer = tf.summary.create_file_writer(f'./logs/test/{self.model_name}')
        checkpoint = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), model=self)

        for epoch in range(num_train_steps):
            np.random.shuffle(list_datapoints)
            avg_loss = 0
            for i in range(num_batches):
                batch_indices = list_datapoints[i * self.batch_size: min((i + 1) * self.batch_size, num_datapoints)]
                self.inputs = traindata[batch_indices, :]
                self.targets = trainlabel[batch_indices]
                self.learning_rate = self.lr_initial * 2 ** (-np.floor(epoch / self.lr_decay_step))
                loss = self.train_step()
                avg_loss += loss / num_batches

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', avg_loss, step=epoch)

            self.inputs = testdata
            self.targets = testlabel
            test_loss = self.compute_loss()
            with test_summary_writer.as_default():
                tf.summary.scalar('test_loss', test_loss, step=epoch)

            checkpoint.save(file_prefix=checkpoint_prefix)

            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train loss {avg_loss:.3f}, Test loss {test_loss:.3f}')

        train_summary_writer.close()
        test_summary_writer.close()

# Example usage:
# Define your data and parameters
# model = Model(inputs, targets, learning_rate, reg_constant, dropout_rate, num_neurons,
#               lr_initial, lr_decay_step, batch_size, model_name)
# model.train(traindata, trainlabel, testdata, testlabel, num_train_steps)
