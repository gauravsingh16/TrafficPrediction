import os
import datetime as dt
import timer as timer
from keras.callbacks import EarlyStopping, ModelCheckpoint

def train(self, x, y, epochs, batch_size, save_dir):
		
		timer.start()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
		
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		callbacks = [
			EarlyStopping(monitor='val_loss', patience=2),
			ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
		]
		self.model.fit(
			x,
			y,
			epochs=epochs,
			batch_size=batch_size,
			callbacks=callbacks
		)
		self.model.save(save_fname)

		print('[Model] Training Completed. Model saved as %s' % save_fname)
		timer.stop()

def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
	timer.start()
	print('[Model] Training Started')
	print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
		
	save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
	callbacks = [
		ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
	]
	self.model.fit_generator(
		data_gen,
		steps_per_epoch=steps_per_epoch,
		epochs=epochs,
		callbacks=callbacks,
		workers=1
	)
		
	print('[Model] Training Completed. Model saved as %s' % save_fname)
	timer.stop()
