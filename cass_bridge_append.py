    def copy_case(self, dest_dir, model_names=('CT Soft Tissue', 'Skull','Mandible','Lower Teeth')):
        
        sub_info = [self._rar_file, *self.subject_info]
        os.makedirs(dest_dir, exist_ok=True)

        for name in model_names:
            if not self.has_model(name):
                print(' '.join((f'  {self._rar_file} is missing models:', name)))
                return None
            
        models = f.load_models(model_names)
            
        for im,m in enumerate(models):

            # transform stl according to cass rule
            v4 = np.hstack((m.v, np.ones((m.v.shape[0],1))))
            v4 = v4 @ np.array(m.T[0]).T
            m.v[...] = v4[:,:3]

            # write stl
            stl_name = model_names[im]
            if stl_name == 'Upper Teeth (original)':
                stl_name = 'Upper Teeth'
            stl_name += '.stl'
            write_stl(m, os.path.join(dest_dir, stl_name))

            # write two transformations
            if model_names[im] == 'Skull':
                with open(os.path.join(dest_dir, 'global_t.tfm'), 'w', newline='') as f:
                    csv.writer(f, delimiter=' ').writerows(m.T[0])

            elif model_names[im] == 'Mandible':
                with open(os.path.join(dest_dir, 'mandible_t.tfm'), 'w', newline='') as f:
                    csv.writer(f, delimiter=' ').writerows(m.T[0])




