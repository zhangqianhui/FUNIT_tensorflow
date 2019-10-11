from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tfLib.ops import *
from tfLib.loss import *
from Dataset import save_images
import os
import numpy as np
import PyLib.const as con

con.EPS = 1e-7
class FSUGAN(object):

    # build model
    def __init__(self, data_ob, opt):

        self.opt = opt
        # placeholder defination
        self.data_ob = data_ob
        self.x = tf.placeholder(tf.float32,[opt.batchSize, opt.image_size, opt.image_size, opt.input_nc])
        self.y_1 = tf.placeholder(tf.float32,[opt.batchSize, opt.image_size, opt.image_size, opt.input_nc])
        self.cls_x = tf.placeholder(tf.int32, [opt.batchSize])
        self.cls_y = tf.placeholder(tf.int32, [opt.batchSize])
        self.lr_decay = tf.placeholder(tf.float32, None, name='lr_decay')

    def build_model(self):

        self.content_code = self.content_encoder(self.x)
        self.encode_y1 = self.class_encoder_k(self.y_1)
        self.class_code = self.encode_y1
        self.tilde_x = self.decoder(content_code=self.content_code, class_code=self.class_code)
        self.encode_x = self.class_encoder_k(self.x)
        self.x_recon = self.decoder(content_code=self.content_code, class_code=self.encode_x)
        self.content_recon_loss = tf.reduce_mean(tf.abs(self.x - self.x_recon))

        self.x_feature, self.D_real_x = self.discriminator(self.x)
        self.y_feature_1, self.D_real_y = self.discriminator(self.y_1)
        self.tilde_x_feature, self.D_fake = self.discriminator(self.tilde_x)
        self.x_feature_recon, self.D_fake_recon = self.discriminator(self.x_recon)

        self.feature_matching = 0.5 * getfeature_matching_loss(self.y_feature_1, self.tilde_x_feature) + \
                            0.5 * getfeature_matching_loss(self.x_feature, self.x_feature_recon)

        self.D_gan_loss = self.loss_hinge_dis(self.D_real_y, self.D_fake, self.cls_y, self.cls_y)
        self.G_gan_loss = 0.5 * self.loss_hinge_gen(self.D_fake, self.cls_y) \
                          + 0.5 * self.loss_hinge_gen(self.D_fake_recon, self.cls_x)
        self.grad_penalty = self.gradient_penalty_just_real(x=self.y_1, label=self.cls_y)

        # weight decay
        self.l2_loss_d = getWeight_Decay(scope='discriminator')
        self.l2_loss_g = getWeight_Decay(scope='content_encoder') + getWeight_Decay(scope='class_encoder_k') + getWeight_Decay(scope='decoder')

        self.D_loss = self.D_gan_loss + self.opt.lam_gp * self.grad_penalty + self.l2_loss_d
        self.G_loss = self.G_gan_loss + self.opt.lam_recon * self.content_recon_loss + self.opt.lam_fp * self.feature_matching + self.l2_loss_g

    def train(self):

        log_vars = []
        log_vars.append(('D_loss', self.D_loss))
        log_vars.append(('G_loss', self.G_loss))

        vars = tf.trainable_variables()

        '''
        total_para = 0
        for variable in vars:
            shape = variable.get_shape()
            print(variable.name, shape)
            variable_para = 1
            for dim in shape:
                variable_para *= dim.value
            total_para += variable_para
        print("The total para", total_para)
        '''

        g_vars = getTrainVariable(vars, scope='encoder') + getTrainVariable(vars, scope='decoder')
        d_vars = getTrainVariable(vars, scope='discriminator')

        assert len(vars) == len(g_vars) + len(d_vars)

        saver = tf.train.Saver()
        for k, v in log_vars:
            tf.summary.scalar(k, v)

        opti_G = tf.train.RMSPropOptimizer(self.opt.lr_g * self.lr_decay).minimize(loss=self.G_loss,
                                                                                          var_list=g_vars)
        opti_D = tf.train.RMSPropOptimizer(self.opt.lr_g * self.lr_decay).minimize(loss=self.D_loss,
                                                                                          var_list=d_vars)
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.opt.log_dir, sess.graph)

            ckpt = tf.train.get_checkpoint_state(self.opt.checkpoints_dir)
            if ckpt and ckpt.model_checkpoint_path:
                start_step = int(ckpt.model_checkpoint_path.split('model_', 2)[1].split('.', 2)[0])
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Load Successfully!', ckpt.model_checkpoint_path)
            else:
                start_step = 0

            step = start_step
            lr_decay = 1

            print("Start reading dataset")
            while step <= self.opt.niter:

                if step > self.opt.niter_decay and step % 2000 == 0:
                    lr_decay = (self.opt.niter - step) / float(self.opt.niter - self.opt.iter_decay)

                source_image_x_data, target_image_y1_data, cls_x, cls_y = self.data_ob.getNextBatch()
                source_image_x = self.data_ob.getShapeForData(source_image_x_data)
                target_image_y1 = self.data_ob.getShapeForData(target_image_y1_data)

                f_d = {
                    self.x :source_image_x, self.y_1:target_image_y1, self.cls_x: cls_x, self.cls_y: cls_y, self.lr_decay: lr_decay
                }

                sess.run(opti_D, feed_dict=f_d)
                sess.run(opti_G, feed_dict=f_d)

                summary_str = sess.run(summary_op, feed_dict=f_d)
                summary_writer.add_summary(summary_str, step)

                if step % self.opt.display_freq == 0:

                    output_loss = sess.run([self.D_loss, self.D_gan_loss, self.G_loss, self.G_gan_loss,
                                             self.content_recon_loss, self.feature_matching, self.l2_loss_d, self.l2_loss_g], feed_dict=f_d)
                    print("step %d, D_loss=%.4f, D_gan_loss=%.4f"
                          " G_loss=%.4f, G_gan_loss=%.4f, content_recon=%.4f, feautre_loss=%.4f, l2_loss=%.4f, lr_decay=%.4f" 
                          % (step, output_loss[0], output_loss[1], output_loss[2], output_loss[3],
                                                          output_loss[4], output_loss[5], output_loss[6] + output_loss[7], lr_decay))

                if np.mod(step, self.opt.save_latest_freq) == 0:

                    f_d = {
                        self.x: source_image_x, self.y_1: target_image_y1}

                    train_output_img = sess.run([
                        self.x,
                        self.y_1,
                        self.tilde_x,
                        self.x_recon
                    ],feed_dict=f_d)

                    output_img = np.concatenate([train_output_img[0],
                                                 train_output_img[1],
                                                 train_output_img[2],
                                                 train_output_img[3]],axis=0)

                    save_images(output_img, [output_img.shape[0]/self.opt.batchSize, self.opt.batchSize],
                                '{}/{:02d}_output_img.jpg'.format(self.opt.sample_dir, step))

                if np.mod(step, self.opt.save_model_freq) == 0 and step != 0:
                    saver.save(sess, os.path.join(self.opt.checkpoints_dir, 'model_{:06d}.ckpt'.format(step)))
                step += 1

            save_path = saver.save(sess, os.path.join(self.opt.checkpoints_dir, 'model_{:06d}.ckpt'.format(step)))
            summary_writer.close()

            print("Model saved in file: %s" % save_path)

    def test(self):
        pass

    def reshape_tile(self, cls_l):
        return tf.tile(tf.reshape(tf.one_hot(cls_l, depth=self.opt.num_source_class),
                shape=[self.opt.batchSize, 1, 1, self.opt.num_source_class]), multiples=[1, 8, 8, 1])

    #conditional hinge loss
    def loss_hinge_dis(self, d_real_logits, d_fake_logits, cls_x, cls_y):
        cls_x = self.reshape_tile(cls_x)
        cls_y = self.reshape_tile(cls_y)
        loss = tf.reduce_mean(tf.nn.relu(tf.reduce_sum(cls_x * (1.0 - d_real_logits), axis=3)))
        loss += tf.reduce_mean(tf.nn.relu(tf.reduce_sum(cls_y * (1.0 + d_fake_logits), axis=3)))

        return loss

    def loss_hinge_gen(self, d_fake_logits, cls_x):
        cls_x = self.reshape_tile(cls_x)
        loss = - tf.reduce_mean(tf.reduce_sum(cls_x * d_fake_logits, axis=3))
        return loss

    def gradient_penalty_just_real(self, x, label):
        label = self.reshape_tile(label)
        _, discri_logits = self.discriminator(x)
        discri_logits = tf.squeeze(tf.reduce_sum(discri_logits * label, axis=3))
        gradients = tf.gradients(tf.reduce_mean(discri_logits), [x])[0]
        slopes = tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3])
        return tf.reduce_mean(slopes)

    def content_encoder(self, x):

        necf_t = self.opt.necf_t
        n_g_ref_t = self.opt.n_g_ref_t
        with tf.variable_scope("content_encoder", reuse=tf.AUTO_REUSE):

            x = conv2d(x, output_dim=necf_t, kernel=7, stride=1, padding='SAME', scope='conv-1')
            x = instance_norm(x,scope='IN-1', affine=False)
            x = tf.nn.relu(x)
            for i in range(self.opt.n_layers_ec):
                x = conv2d(x, output_dim=pow(2,i+1)* necf_t, kernel=4, stride=2, padding='SAME', scope='conv_{}'.format(i+1))
                x = instance_norm(x,scope='ins_{}'.format(i+1), affine=False)
                x = tf.nn.relu(x)

            for i in range(2):
                x = Resblock(x, channels=n_g_ref_t, is_start=False, is_norm=True, is_acti=True, affline=False, scope='residual_{}'.format(i))

        return x

    def class_encoder_k(self, y):

        nesf_t = self.opt.nesf_t
        with tf.variable_scope("class_encoder_k", reuse=tf.AUTO_REUSE):
            y = tf.nn.relu(conv2d(y, output_dim=nesf_t, kernel=7, stride=1, padding='SAME',scope='conv-1'))
            for i in range(2):
                y = conv2d(y, output_dim=nesf_t * pow(2, i+1), kernel=4, stride=2, padding='SAME',scope='conv_{}'.format(i+1))
                y = tf.nn.relu(y)
            for i in range(self.opt.n_layers_es - 2):
                y = conv2d(y, output_dim=nesf_t * pow(2, 2), kernel=4, stride=2, padding='SAME',scope='conv_{}'.format(i+3))
                y = tf.nn.relu(y)
            y = Adaptive_pool2d(y, output_size=1)
            y = conv2d(y, output_dim=nesf_t, kernel=1, stride=1, padding='SAME')

        return tf.squeeze(y)

    def decoder(self, content_code, class_code):

        n_g_ref_t = self.opt.n_g_ref_t
        output_nc = self.opt.output_nc
        n_layers_de = self.opt.n_layers_de
        n_residual_de = self.opt.n_residual_de
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):

            #MLP
            for i in range(3):
                if i == 2:
                    class_code = fully_connect(input_=class_code, output_size=n_g_ref_t*8, scope='fc_{}'.format(i+1))
                else:
                    class_code = tf.nn.relu(fully_connect(input_=class_code, output_size=n_g_ref_t // 2, scope='fc_{}'.format(i+1)))

            de = content_code

            for i in range(n_residual_de):
                mean1 = class_code[:,n_g_ref_t*i:n_g_ref_t*(i+1)]
                stand_dev1 = class_code[:,n_g_ref_t*(i+1):n_g_ref_t*(i+2)]
                mean2 = class_code[:,n_g_ref_t*(i+2):n_g_ref_t*(i+3)]
                stand_dev2 = class_code[:,n_g_ref_t*(i+3):n_g_ref_t*(i+4)]
                print(class_code)
                class_code = class_code[:,n_g_ref_t*(i+3):]

                de = Resblock_AdaIn(content_code, beta1=mean1, gamma1=stand_dev1, beta2=mean2, gamma2=stand_dev2,
                                    channels=n_g_ref_t, scope='res_{}'.format(i+1))

            n_g_ref_t = n_g_ref_t // 2
            for i in range(n_layers_de):
                de = upscale(de, scale=2)
                de = conv2d(de, output_dim=n_g_ref_t/pow(2,i), kernel=5, stride=1, padding='SAME', scope='conv_{}'.format(i+1))
                de = instance_norm(de, scope='ins_{}'.format(i+1), affine=False)
                de = tf.nn.relu(de)

            y = conv2d(de, output_dim=output_nc, kernel=7, stride=1, padding='SAME', scope='conv_final')

            return tf.nn.tanh(y)

    def discriminator(self, x):

        ndf = self.opt.ndf
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):

            x = lrelu(conv2d(input_=x, output_dim=ndf, kernel=7, stride=1, scope='conv-64'))
            for i in range(5):
                if i == 4:
                    x = Resblock_D(x, channels=min(ndf * pow(2, i+1), 1024), is_acti=False, is_start=False, is_norm=False, scope='r1_{}'.format(i + 1))
                    x = Resblock_D(x, channels=min(ndf * pow(2, i+1), 1024), is_acti=False, is_start=False, is_norm=False, scope='r2_{}'.format(i+1))
                else:
                    x = Resblock_D(x, channels=min(ndf * pow(2, i+1), 1024), is_acti=True, is_start=True,
                                   is_norm=False,
                                   scope='r1_{}'.format(i + 1))
                    x = Resblock_D(x, channels=min(ndf * pow(2, i+1), 1024), is_acti=True, is_start=False, is_norm=False, scope='r2_{}'.format(i+1))
                    x = avgpool2d(x, k=2)

            x_predict = conv2d(lrelu(x), output_dim=self.opt.num_source_class, kernel=3, stride=1, padding='SAME')

        return x, tf.squeeze(x_predict)









