/*
 * Description:  Class implementing the SSH protocol
 * TODO:         comment (fabien): the SSH class should not have any DARwIn-OP dependency
 *               -> SSH stuff is not reusable
 *               -> the code is difficult to understand because mixed
 */


#ifndef SSH_HPP
#define SSH_HPP

#include <libssh/libssh.h>
#include <libssh/sftp.h>
#include <libssh/callbacks.h>

#include <QtCore/QString>
#include <QtCore/QObject>

#include "common.hpp"

class SSH : public QObject {
  Q_OBJECT

  public:
                  SSH(QObject *parent, RobotisModel robotisModel);
    virtual      ~SSH();

    int           startRemoteCompilation(const QString &IPAddress, const QString &username, const QString &password, bool makeDefaultController);
    int           startRemoteControl(const QString &IPAddress, const QString &username, const QString &password);
    int           uninstall(const QString &IPAddress, const QString &username, const QString &password);
    void          terminate() { mTerminate = true; }
    const QString error();

  signals:
    void          print(const QString &message, bool error);
    void          status(const QString &message);
    void          done();

  protected:
    QString       mError;
    QString       mStdout;
    QString       mStderr;

  private:
    int           openSSHSession(const QString &IPAddress, const QString &username, const QString &password);
    void          closeSSHSession();
    int           openSSHChannel();
    int           openSFTPChannel();
    void          closeSSHChannel();
    void          closeSFTPChannel();
    int           readRemoteFile(const QString &fileName, char *buffer, int buffer_size);
    int           verifyKnownHost();
    int           sendFile(const QString &source, const QString &target);
    int           executeSSHCommand(const QString &command, bool display = true, bool wait = true);
    void          readChannel(bool display, int err);
    int           updateFrameworkIfNeeded(const QString &username, const QString &rootPassword);
    bool          isFrameworkUpToDate();
    int           updateFramework(const QString &username, const QString &rootPassword);
    int           updateWrapperIfNeeded(const QString &username, const QString &rootPassword);
    bool          isWrapperUpToDate();
    int           updateWrapper(const QString &username, const QString &rootPassword);
    QString       mainDirectoryName();
    QString       makeVars();
    void          log(const QString &message, bool error = false);
    void          info(const QString &message);

    ssh_session   mSSHSession;
    ssh_channel   mSSHChannel;
    sftp_session  mSFTPChannel;
    sftp_file     mSFTPFile;
    bool          mTerminate;
    RobotisModel   mActualModel;
};

#endif
