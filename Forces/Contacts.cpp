
#include "Contacts.cuh"
#include "GPU_contacts.cuh"

void CustomContactInteractor::anchor() {}

CustomContactInteractor::~CustomContactInteractor(){
    if (d_contact_mgr) {
            cudaFree(d_contact_mgr);
            d_contact_mgr = nullptr;
        }
}

void CustomContactInteractor::updateSimulationTime(real newTime) override { time = newTime; }
void CustomContactInteractor::h_cleanupContacts(cudaStream_t st){
    // std::cout << "Contact cleanup check at time " << time << std::endl;
    // Periodically clean up inactive contacts 
    ++steps; 
    if ((steps % 10000) == 0) {
        std::cout << "DEBUG step=" << steps
                << " time/dt=" << (time/dt)
                << " time=" << time << "\n";
    }
    if ( (steps % cleanup_interval) == 0){
        cudaStreamSynchronize(st);
        auto policy = thrust::cuda::par.on(st);

        thrust::device_ptr<ContactHistory> d_begin(contact_mgr->contacts);
        thrust::device_ptr<ContactHistory> d_end = d_begin + contact_mgr->max_contacts; 
        int h_count = thrust::count_if(
        policy,
        d_begin,
        d_end, 
        [] __device__ (const ContactHistory& contact) { return contact.is_active; }
        );

        // int h_count = 0; 
        // cudaMemcpy(&h_count, contact_mgr->contact_count, sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "Starting contact cleanup at time " << time << " with " << h_count << " active contacts." << std::endl;
        // fprintf(stdout, "Cleaning up inactive contacts at time %f (step %d)\n", time, steps);
        int threads = 128;
        int blocks = (contact_mgr->max_contacts + threads - 1) / threads;
        gpu_cleanupContacts<<<blocks, threads, 0, st>>>(d_contact_mgr, contact_mgr->hash_table, time-2*dt);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "Device sync failed after kernel: %s\n", cudaGetErrorString(err));
            // Optional: exit or raise error so program stops near the failing kernel
            exit(1);
        }
        // Print number of cleaned contacts
        // cudaMemcpy(&h_count, contact_mgr->cleanup_count, sizeof(int), cudaMemcpyDeviceToHost);
        // std::cout << "Cleaned up " << h_count << " inactive contacts at time " << time << std::endl;

        int cleaned;
        cudaMemcpy(&cleaned, contact_mgr->cleanup_count, sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "Marked " << cleaned << " contacts as inactive" << std::endl;

        int total_active = count_if(
        policy,
        d_begin,
        d_end, 
        [] __device__ (const ContactHistory& contact) { return contact.is_active; }
        );
        std::cout << "Total active contacts before compaction: " << total_active << std::endl;

        // Reset cleanup count
        // int zero = 0;
        // cudaMemcpy(contact_mgr->cleanup_count, &zero, sizeof(int), cudaMemcpyHostToDevice);

        // Copy contact data back to host for compaction
        // cudaMemcpy(contact_mgr->contacts, d_contact_mgr->contacts, 
        //            contact_mgr->max_contacts * sizeof(ContactHistory), 
        //            cudaMemcpyDeviceToHost);
        // cudaMemcpy(contact_mgr->contact_count, d_contact_mgr->contact_count, 
        //            sizeof(int), 
        //            cudaMemcpyDeviceToHost);

        std::cout << "Compacting contacts array after cleanup..." << std::endl;
        // thrust::device_ptr<ContactHistory> d_begin(contact_mgr->contacts);
        // thrust::device_ptr<ContactHistory> d_end = d_begin + contact_mgr->max_contacts;  
        // Compact the contacts array to remove inactive contacts
        // auto new_end = thrust::remove_if(
        //   thrust::device,
        //   contact_mgr->contacts,
        //   contact_mgr->contacts + cleaned,
        //   [] __device__ (const ContactHistory& contact) { return !contact.is_active; }
        // );

        auto new_end = thrust::partition(
        // thrust::device,
        policy,
        d_begin,
        d_end,
        [] __device__ (const ContactHistory& contact) { return contact.is_active; }
        );
        std::cout << "Compaction complete." << std::endl;
        // Obtain new contact count by summing over active contacts
        int new_count = thrust::distance(d_begin, new_end);
        // int new_count = thrust::count_if(
        //   d_begin,
        //   new_end, 
        //   [] __device__ (const ContactHistory& contact) { return contact.is_active; }
        // );
        std::cout << "After compaction: " << new_count << " active contacts (removed " 
                << (h_count - new_count) << ")" << std::endl;

        // int new_count = thrust::count_if(
        //     thrust::device, 
        //     contact_mgr->contacts, 
        //     contact_mgr->contacts + ((contact_mgr->max_contacts)), 
        //     [] __device__ (const ContactHistory& contact) { return contact.is_active; }
        //   );
        // cudaMemcpy(contact_mgr->contact_count, &new_count, sizeof(int), cudaMemcpyHostToDevice);
        // std::cout << "Updated contact count after compaction: " << new_count << std::endl;

        // Step 4: Update contact counts
        cudaMemset(contact_mgr->contact_count, 0, sizeof(int));
        cudaMemcpy(contact_mgr->contact_count, &new_count, sizeof(int), 
        cudaMemcpyHostToDevice);

        // Reset cleanup counter
        cudaMemset(contact_mgr->cleanup_count, 0, sizeof(int));

        // Copy updated contact data back to device
        // cudaMemcpy(d_contact_mgr->contacts, contact_mgr->contacts, 
        //            contact_mgr->max_contacts * sizeof(ContactHistory), 
        //            cudaMemcpyHostToDevice);
        // cudaMemcpy(contact_mgr->contact_count, &new_count, sizeof(int), cudaMemcpyHostToDevice);
        
        // Reset the hash table 
        contact_mgr->resetHashTable();
        blocks = (new_count + threads - 1) / threads;
        std::cout << "Rebuilding hash table for " << new_count << " contacts..." << std::endl;
        
        gpu_rebuildHashTable<<<blocks, threads, 0, st>>>(d_contact_mgr, contact_mgr->hash_table);
        
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "Device sync failed after kernel: %s\n", cudaGetErrorString(err));
            // Optional: exit or raise error so program stops near the failing kernel
            exit(1);
        }

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Rebuild kernel failed: %s\n", cudaGetErrorString(err));
            exit(1);
        }
        // cudaStreamSynchronize(st);

        std::cout << "Cleanup complete.\n" << std::endl;
        // std::cout << "Rebuilding hash table after cleanup..." << std::endl;
        // h_rebuildHashTable(st);

    }
    return;
}

void CustomContactInteractor::sum(Computables comp, cudaStream_t st) {
    // std::cout << "=========" << " CustomContactInteractor at time " << time << " " <<"=========" << std::endl;
    Box box(boxSize);
    nl->update(box, rcut, st);
    comp.torque = false; // Ensure torque is computed
    comp.stress = true; // Ensure stress is computed

    // DON'T mark contacts inactive - we want to preserve history!
    // Instead, the kernel will mark contacts as active when found
    
    // NeighbourContainer can provide forward iterators with the neighbours of
    // each particle The drawback of it being a forward iterator is that it can
    // only be advanced, once you have asked for the next neighbour there is no
    // going back without starting from the first. With it=ni.begin() you can
    // only do it++, etc, there is no operator[] nor it--
    auto ni = nl->getNeighbourContainer();
    auto vel = pd->getHalfVel(access::gpu, access::read).raw();
    auto ang_vel = pd->getHalfAngVel(access::gpu, access::read).raw();
    auto radius = pd->getRadius(access::gpu, access::read).raw();
    auto force =
        comp.force
            ? pd->getForce(access::location::gpu, access::mode::readwrite).raw()
            : nullptr;

    auto torque = 
        comp.torque
            ? pd->getTorque(access::location::gpu, access::mode::readwrite).raw()
            : nullptr;

    auto stress_x = 
        comp.stress
            ? pd->getStressX(access::location::gpu, access::mode::readwrite).raw()
            : nullptr;

    auto stress_y =
        comp.stress
            ? pd->getStressY(access::location::gpu, access::mode::readwrite).raw()
            : nullptr;

    auto stress_z =
        comp.stress
            ? pd->getStressZ(access::location::gpu, access::mode::readwrite).raw()
            : nullptr;

    auto energy = comp.energy ? pd->getEnergy(access::location::gpu,
                                              access::mode::readwrite)
                                    .raw()
                              : nullptr;
    auto virial = comp.virial ? pd->getVirial(access::location::gpu,
                                              access::mode::readwrite)
                                    .raw()
                              : nullptr;
    processNeighboursContacts<decltype(ni)><<<numberParticles / 128 + 1, 128, 0, st>>>(
        ni, time, Nwrite, numberParticles, box, radius, vel, ang_vel, force, torque, energy, virial, stress_x, stress_y, stress_z,
        dt, kn, kt, mu, gamma_n, gamma_t, d_contact_mgr);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Device sync failed after kernel: %s\n", cudaGetErrorString(err));
        // Optional: exit or raise error so program stops near the failing kernel
        exit(1);
    }

    h_cleanupContacts(st);

};